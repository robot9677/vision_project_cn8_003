import json
import os
import time
from datetime import datetime
from typing import Any, Dict, Optional


ERROR_NAMES = {
    0: "NO_ERROR",

    10: "CAMERA_INIT_FAILED",
    11: "CAMERA_FRAME_TIMEOUT",

    20: "LIGHT_START_FAILED",
    21: "LIGHT_COMMUNICATION_FAILED",

    30: "RECIPE_LOAD_FAILED",
    31: "ROI_CONFIG_ERROR",

    40: "INSPECTION_FAILED",
    41: "INSPECTION_EXCEPTION",

    50: "COMMAND_SEQUENCE_ERROR",
    51: "EMERGENCY_STOP",

    60: "VISION_RESET_FAILED",

    70: "PLC_SERIAL_OPEN_FAILED",
    71: "PLC_SERIAL_COMMUNICATION_LOST",

    90: "SHUTDOWN_FAILED",
}


_DEFAULT_DIAGNOSTICS_CONFIG: Dict[str, Any] = {
    "enabled": True,
    "index_path": "diagnostics/diagnostics_index.jsonl",
    "index_max_bytes": 2 * 1024 * 1024,
    "index_keep_files": 2,
    "keep_days": 14,
    "error_keep_files": 60,
    "test_keep_files": 120,
    "normal_keep_files": 5,
    "normal_sample_every": 20,
    "prune_interval_sec": 30.0,
}

_DIAGNOSTICS_CONFIG: Dict[str, Any] = dict(_DEFAULT_DIAGNOSTICS_CONFIG)
_DIAGNOSTICS_LOGS_ROOT = ""
_LAST_PRUNE_EPOCH = 0.0


def get_error_name(error_code: int) -> str:
    return ERROR_NAMES.get(int(error_code), "UNDEFINED_ERROR")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return str(value)


def configure_diagnostics(
    config: Optional[Dict[str, Any]],
    logs_root: str,
) -> Dict[str, Any]:
    """Configure diagnostic indexing and retention.

    This only manages debug files. It does not affect inspection, PLC, camera,
    or light control behavior.
    """
    global _DIAGNOSTICS_CONFIG, _DIAGNOSTICS_LOGS_ROOT

    merged = dict(_DEFAULT_DIAGNOSTICS_CONFIG)
    if isinstance(config, dict):
        merged.update(config)

    merged["enabled"] = bool(merged.get("enabled", True))
    merged["index_max_bytes"] = max(
        256 * 1024,
        int(merged.get("index_max_bytes", 2 * 1024 * 1024)),
    )
    merged["index_keep_files"] = max(
        1,
        int(merged.get("index_keep_files", 2)),
    )
    merged["keep_days"] = max(0, int(merged.get("keep_days", 14)))
    merged["error_keep_files"] = max(
        1,
        int(merged.get("error_keep_files", 60)),
    )
    merged["test_keep_files"] = max(
        1,
        int(merged.get("test_keep_files", 120)),
    )
    merged["normal_keep_files"] = max(
        1,
        int(merged.get("normal_keep_files", 5)),
    )
    merged["normal_sample_every"] = max(
        1,
        int(merged.get("normal_sample_every", 20)),
    )
    merged["prune_interval_sec"] = max(
        1.0,
        float(merged.get("prune_interval_sec", 30.0)),
    )

    _DIAGNOSTICS_CONFIG = merged
    _DIAGNOSTICS_LOGS_ROOT = os.path.abspath(logs_root)

    if merged["enabled"]:
        os.makedirs(_resolve_index_dir(_DIAGNOSTICS_LOGS_ROOT), exist_ok=True)
        prune_diagnostic_logs(_DIAGNOSTICS_LOGS_ROOT, force=True)

    return dict(_DIAGNOSTICS_CONFIG)


def get_diagnostics_config() -> Dict[str, Any]:
    return dict(_DIAGNOSTICS_CONFIG)


def _resolve_index_path(logs_root: str) -> str:
    path = str(
        _DIAGNOSTICS_CONFIG.get(
            "index_path",
            "diagnostics/diagnostics_index.jsonl",
        )
    ).strip()
    if os.path.isabs(path):
        return os.path.abspath(path)
    return os.path.abspath(os.path.join(logs_root, path))


def _resolve_index_dir(logs_root: str) -> str:
    return os.path.dirname(_resolve_index_path(logs_root))


def _rotate_file(path: str, keep_files: int):
    for index in range(int(keep_files), 0, -1):
        src = path if index == 1 else f"{path}.{index - 1}"
        dst = f"{path}.{index}"
        if not os.path.exists(src):
            continue
        try:
            if os.path.exists(dst):
                os.remove(dst)
            os.replace(src, dst)
        except Exception:
            pass


def _append_diagnostic_index(
    logs_root: str,
    kind: str,
    path: str,
    summary: Dict[str, Any],
):
    if not bool(_DIAGNOSTICS_CONFIG.get("enabled", True)):
        return

    index_path = _resolve_index_path(logs_root)
    os.makedirs(os.path.dirname(index_path), exist_ok=True)

    try:
        if (
            os.path.exists(index_path)
            and os.path.getsize(index_path)
            >= int(_DIAGNOSTICS_CONFIG.get("index_max_bytes", 2 * 1024 * 1024))
        ):
            _rotate_file(
                index_path,
                int(_DIAGNOSTICS_CONFIG.get("index_keep_files", 2)),
            )

        record = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "epoch": time.time(),
            "kind": str(kind),
            "path": os.path.relpath(path, logs_root),
        }
        record.update(_json_safe(summary or {}))

        with open(index_path, "a", encoding="utf-8", buffering=1) as file:
            file.write(
                json.dumps(record, ensure_ascii=False, separators=(",", ":"))
                + "\n"
            )
    except Exception as index_error:
        print(f"[DIAGNOSTICS] index write failed: {index_error}")


def _collect_files(root: str):
    items = []
    if not os.path.isdir(root):
        return items

    for dir_path, _, filenames in os.walk(root):
        for filename in filenames:
            if not filename.lower().endswith(".json"):
                continue
            path = os.path.join(dir_path, filename)
            try:
                items.append((os.path.getmtime(path), path))
            except OSError:
                pass
    items.sort(reverse=True)
    return items


def _remove_empty_dirs(root: str):
    if not os.path.isdir(root):
        return
    for dir_path, dir_names, _ in os.walk(root, topdown=False):
        for dir_name in dir_names:
            path = os.path.join(dir_path, dir_name)
            try:
                if not os.listdir(path):
                    os.rmdir(path)
            except OSError:
                pass


def _prune_json_tree(root: str, max_files: int, keep_days: int):
    items = _collect_files(root)
    cutoff = None
    if keep_days > 0:
        cutoff = time.time() - (float(keep_days) * 86400.0)

    for index, (mtime, path) in enumerate(items):
        remove = index >= int(max_files)
        if cutoff is not None and mtime < cutoff:
            remove = True
        if not remove:
            continue
        try:
            os.remove(path)
        except OSError:
            pass

    _remove_empty_dirs(root)


def prune_diagnostic_logs(logs_root: str, force: bool = False):
    global _LAST_PRUNE_EPOCH

    if not bool(_DIAGNOSTICS_CONFIG.get("enabled", True)):
        return

    now = time.time()
    interval = float(_DIAGNOSTICS_CONFIG.get("prune_interval_sec", 30.0))
    if not force and (now - _LAST_PRUNE_EPOCH) < interval:
        return
    _LAST_PRUNE_EPOCH = now

    try:
        keep_days = int(_DIAGNOSTICS_CONFIG.get("keep_days", 14))
        _prune_json_tree(
            os.path.join(logs_root, "plc_errors"),
            int(_DIAGNOSTICS_CONFIG.get("error_keep_files", 60)),
            keep_days,
        )
        _prune_json_tree(
            os.path.join(logs_root, "plc_tests"),
            int(_DIAGNOSTICS_CONFIG.get("test_keep_files", 120)),
            keep_days,
        )
        _prune_json_tree(
            os.path.join(logs_root, "normal_reference"),
            int(_DIAGNOSTICS_CONFIG.get("normal_keep_files", 5)),
            keep_days,
        )
    except Exception as prune_error:
        print(f"[DIAGNOSTICS] prune failed: {prune_error}")


def save_plc_error_log(
    logs_root: str,
    event_type: str,
    error_code: int,
    message: str,
    plc_snapshot: Optional[Dict[str, Any]] = None,
    vision_snapshot: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> Optional[str]:

    now = datetime.now()

    log_dir = os.path.join(
        logs_root,
        "plc_errors",
        now.strftime("%Y%m%d"),
    )

    try:
        os.makedirs(log_dir, exist_ok=True)

        timestamp_text = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        filename_time = now.strftime("%H%M%S_%f")

        error_code = int(error_code)

        log_data = {
            "schema_version": 2,
            "timestamp": timestamp_text,
            "event_type": str(event_type),
            "error": {
                "code": error_code,
                "name": get_error_name(error_code),
                "message": str(message or ""),
            },
            "plc_state": _json_safe(plc_snapshot or {}),
            "vision_state": _json_safe(vision_snapshot or {}),
            "exception": None,
        }

        if exception is not None:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "repr": repr(exception),
            }

        filename = (
            f"{filename_time}_"
            f"{error_code:02d}_"
            f"{get_error_name(error_code)}.json"
        )

        final_path = os.path.join(log_dir, filename)
        temp_path = final_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                log_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_path, final_path)

        _append_diagnostic_index(
            logs_root,
            kind="error",
            path=final_path,
            summary={
                "event_type": str(event_type),
                "error_code": error_code,
                "error_name": get_error_name(error_code),
                "message": str(message or ""),
            },
        )
        prune_diagnostic_logs(logs_root)

        print(f"[PLC ERROR LOG] saved: {final_path}")
        return final_path

    except Exception as log_error:
        print(f"[PLC ERROR LOG] save failed: {log_error}")
        return None


def save_plc_test_log(
    logs_root: str,
    test_id: str,
    test_type: str,
    phase: str,
    result: str,
    error_code: int,
    message: str,
    plc_snapshot: Optional[Dict[str, Any]] = None,
    vision_snapshot: Optional[Dict[str, Any]] = None,
    extra: Optional[Dict[str, Any]] = None,
) -> Optional[str]:
    """Save one PLC forced-error test lifecycle record."""
    now = datetime.now()
    log_dir = os.path.join(
        logs_root,
        "plc_tests",
        now.strftime("%Y%m%d"),
    )

    try:
        os.makedirs(log_dir, exist_ok=True)
        payload = {
            "schema_version": 2,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "test_id": str(test_id or ""),
            "test_type": str(test_type or ""),
            "phase": str(phase or ""),
            "result": str(result or ""),
            "error": {
                "code": int(error_code),
                "name": get_error_name(int(error_code)),
                "message": str(message or ""),
            },
            "plc_state": _json_safe(plc_snapshot or {}),
            "vision_state": _json_safe(vision_snapshot or {}),
            "extra": _json_safe(extra or {}),
        }

        safe_type = str(test_type or "unknown").replace("/", "_")
        filename = (
            f"{now.strftime('%H%M%S_%f')}_"
            f"{safe_type}_{str(phase or 'event').lower()}.json"
        )
        final_path = os.path.join(log_dir, filename)
        temp_path = final_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, final_path)

        _append_diagnostic_index(
            logs_root,
            kind="test",
            path=final_path,
            summary={
                "test_id": str(test_id or ""),
                "test_type": str(test_type or ""),
                "phase": str(phase or ""),
                "result": str(result or ""),
                "error_code": int(error_code),
                "message": str(message or ""),
            },
        )
        prune_diagnostic_logs(logs_root)

        print(f"[PLC TEST LOG] saved: {final_path}")
        return final_path
    except Exception as log_error:
        print(f"[PLC TEST LOG] save failed: {log_error}")
        return None


def save_normal_reference_log(
    logs_root: str,
    plc_snapshot: Optional[Dict[str, Any]],
    vision_snapshot: Optional[Dict[str, Any]],
    inspection_summary: Optional[Dict[str, Any]],
) -> Optional[str]:
    """Save a compact, sampled good-inspection reference.

    No raw or overlay image is saved here. The small JSON samples are retained
    only for comparison when diagnosing a later incident.
    """
    if not bool(_DIAGNOSTICS_CONFIG.get("enabled", True)):
        return None

    now = datetime.now()
    log_dir = os.path.join(
        logs_root,
        "normal_reference",
        now.strftime("%Y%m%d"),
    )

    try:
        os.makedirs(log_dir, exist_ok=True)
        payload = {
            "schema_version": 1,
            "timestamp": now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3],
            "plc_state": _json_safe(plc_snapshot or {}),
            "vision_state": _json_safe(vision_snapshot or {}),
            "inspection": _json_safe(inspection_summary or {}),
        }
        final_path = os.path.join(
            log_dir,
            f"{now.strftime('%H%M%S_%f')}_normal_reference.json",
        )
        temp_path = final_path + ".tmp"
        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(payload, file, ensure_ascii=False, indent=2)
        os.replace(temp_path, final_path)

        _append_diagnostic_index(
            logs_root,
            kind="normal_reference",
            path=final_path,
            summary={
                "overall_ok": bool(
                    (inspection_summary or {}).get("overall_ok", False)
                ),
                "elapsed_ms": int(
                    (inspection_summary or {}).get("elapsed_ms", 0)
                ),
            },
        )
        prune_diagnostic_logs(logs_root, force=True)
        return final_path
    except Exception as log_error:
        print(f"[NORMAL REFERENCE] save failed: {log_error}")
        return None
