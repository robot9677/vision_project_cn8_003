#!/usr/bin/env python3
"""Inject a controlled PLC/vision error into a running main_vp.py process.

The tool never opens the RS485 port. It writes one atomic JSON request that
main_vp.py consumes from its normal loop. Use only while error_test.enabled=true.
"""

import argparse
import json
import os
import sys
import time
import uuid
from pathlib import Path
from typing import Any, Dict, Optional


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_CANDIDATES = [
    PROJECT_ROOT / "data" / "config" / "plc_config.json",
    PROJECT_ROOT / "data" / "plc_config.json",
]
DEFAULT_REQUEST_PATH = (
    PROJECT_ROOT / "data" / "runtime" / "plc_error_test_request.json"
)
ERROR_TYPES = {
    "camera": "Code 11: camera error; D200=3 restarts camera",
    "light": "Code 21: light error; D200=3 restarts light",
    "inspection": "Code 40: inspection error; D200=3 resets runtime",
    "plc_comm": (
        "Code 71: logical communication-error state; "
        "serial stays open so D200=3 can be tested"
    ),
}


def _load_config() -> Dict[str, Any]:
    for path in CONFIG_CANDIDATES:
        if not path.exists():
            continue
        with path.open("r", encoding="utf-8-sig") as file:
            cfg = json.load(file)
        if not isinstance(cfg, dict):
            raise RuntimeError(f"invalid PLC config object: {path}")
        cfg["_loaded_path"] = str(path)
        return cfg
    return {}


def _resolve_request_path(cfg: Dict[str, Any]) -> Path:
    test_cfg = cfg.get("error_test", {}) or {}
    raw = str(
        test_cfg.get(
            "request_path",
            "data/runtime/plc_error_test_request.json",
        )
    ).strip()

    path = Path(raw)
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _latest_error_log() -> Optional[Path]:
    root = PROJECT_ROOT / "data" / "logs" / "plc_errors"
    if not root.exists():
        return None

    files = [
        path
        for path in root.rglob("*.json")
        if path.is_file()
    ]
    if not files:
        return None
    return max(files, key=lambda path: path.stat().st_mtime)


def show_status(cfg: Dict[str, Any], request_path: Path):
    test_cfg = cfg.get("error_test", {}) or {}
    enabled = bool(test_cfg.get("enabled", False))

    print("===== PLC ERROR TEST STATUS =====")
    print(f"Project root : {PROJECT_ROOT}")
    print(f"Config       : {cfg.get('_loaded_path', 'NOT FOUND')}")
    print(f"Test enabled : {enabled}")
    print(f"Request path : {request_path}")
    print(
        f"Pending      : "
        f"{'YES' if request_path.exists() else 'NO'}"
    )

    latest = _latest_error_log()
    print(f"Latest log   : {latest or 'NONE'}")

    if latest is not None:
        try:
            with latest.open("r", encoding="utf-8") as file:
                data = json.load(file)
            error = data.get("error", {}) or {}
            print(
                f"  timestamp  : {data.get('timestamp', '')}\n"
                f"  event_type : {data.get('event_type', '')}\n"
                f"  code/name  : {error.get('code')} / "
                f"{error.get('name', '')}\n"
                f"  message    : {error.get('message', '')}"
            )
        except Exception as e:
            print(f"  read error : {e}")


def inject_error(
    cfg: Dict[str, Any],
    request_path: Path,
    error_type: str,
    message: str,
):
    test_cfg = cfg.get("error_test", {}) or {}
    if not bool(test_cfg.get("enabled", False)):
        raise RuntimeError(
            "plc_config.json error_test.enabled is false"
        )

    allowed = {
        str(item).strip().lower()
        for item in (test_cfg.get("allowed_types", ERROR_TYPES.keys()) or [])
    }
    if error_type not in allowed:
        raise RuntimeError(
            f"error type is not allowed by config: {error_type}"
        )

    request_path.parent.mkdir(parents=True, exist_ok=True)

    request_id = (
        f"{int(time.time() * 1000)}-"
        f"{uuid.uuid4().hex[:8]}"
    )
    payload = {
        "schema_version": 1,
        "request_id": request_id,
        "created_at": time.strftime("%Y-%m-%d %H:%M:%S"),
        "type": error_type,
        "message": str(message or ""),
        "source": "plc_error_test.py",
    }

    temp_path = request_path.with_suffix(
        request_path.suffix + ".tmp"
    )
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())

    os.replace(temp_path, request_path)

    print("===== PLC ERROR TEST REQUEST SENT =====")
    print(f"Type       : {error_type}")
    print(f"Meaning    : {ERROR_TYPES[error_type]}")
    print(f"Request ID : {request_id}")
    print(f"Path       : {request_path}")
    print("")
    print("Expected:")
    print("  1. PLC Live Monitor shows D201=3 and error code/detail.")
    print("  2. A JSON error log is saved under data/logs/plc_errors/.")
    print("  3. PLC sends D200=3, then Vision attempts recovery.")
    print("  4. Recovery success: D201=0, D202=0, error cleared.")


def interactive_choice() -> str:
    items = list(ERROR_TYPES.items())
    print("===== PLC FORCED ERROR TEST =====")
    for index, (key, description) in enumerate(items, start=1):
        print(f"{index}. {key:10s} - {description}")
    print("q. quit")

    while True:
        value = input("Select: ").strip().lower()
        if value == "q":
            raise KeyboardInterrupt
        if value in ERROR_TYPES:
            return value
        try:
            index = int(value)
        except ValueError:
            continue
        if 1 <= index <= len(items):
            return items[index - 1][0]


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Inject a controlled error into running main_vp.py "
            "without opening the RS485 port."
        )
    )
    parser.add_argument(
        "type",
        nargs="?",
        choices=sorted(ERROR_TYPES),
        help="error type; omit for interactive menu",
    )
    parser.add_argument(
        "--message",
        default="",
        help="optional custom message",
    )
    parser.add_argument(
        "--status",
        action="store_true",
        help="show test configuration and latest error log",
    )
    args = parser.parse_args()

    cfg = _load_config()
    request_path = _resolve_request_path(cfg)

    if args.status:
        show_status(cfg, request_path)
        return

    error_type = args.type or interactive_choice()
    inject_error(
        cfg=cfg,
        request_path=request_path,
        error_type=error_type,
        message=args.message,
    )


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("\n[PLC ERROR TEST] cancelled")
    except Exception as e:
        print(f"[PLC ERROR TEST] failed: {e}")
        sys.exit(1)
