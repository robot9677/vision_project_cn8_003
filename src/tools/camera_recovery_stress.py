#!/usr/bin/env python3
"""Repeatedly inject a capture-worker stall and verify automatic recovery.

Run while main_vp.py is running in RUN mode.  This does not disconnect the
physical camera; it pauses the isolated capture worker so the same stale-frame
monitor and recovery path used for a real Argus stall is exercised.
"""

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
HW_CONFIG = PROJECT_ROOT / "data" / "config" / "hardware_config.json"


def load_paths():
    with HW_CONFIG.open("r", encoding="utf-8-sig") as file:
        cfg = json.load(file)
    active = cfg.get("active_camera_set")
    cam_set = (cfg.get("camera_sets") or {}).get(active) or {}
    cameras = cam_set.get("cameras") or []
    if not cameras:
        raise RuntimeError("active camera config is missing")
    process_cfg = cameras[0].get("process_isolation") or {}
    status = Path(process_cfg.get("status_path", "data/runtime/camera_process_status.json"))
    request = Path(process_cfg.get("request_path", "data/runtime/camera_process_request.json"))
    if not status.is_absolute():
        status = PROJECT_ROOT / status
    if not request.is_absolute():
        request = PROJECT_ROOT / request
    return status, request


def read_json(path: Path):
    with path.open("r", encoding="utf-8") as file:
        return json.load(file)


def atomic_write_json(path: Path, payload):
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    with temp.open("w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)
        file.flush()
        os.fsync(file.fileno())
    os.replace(str(temp), str(path))


def wait_until(predicate, timeout_sec: float, poll_sec: float = 0.1):
    deadline = time.monotonic() + timeout_sec
    last = None
    while time.monotonic() < deadline:
        try:
            last = predicate()
            if last:
                return last
        except (FileNotFoundError, json.JSONDecodeError):
            pass
        time.sleep(poll_sec)
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cycles", type=int, default=10)
    parser.add_argument("--interval", type=float, default=10.0)
    parser.add_argument("--timeout", type=float, default=35.0)
    args = parser.parse_args()

    cycles = max(1, args.cycles)
    interval = max(0.0, args.interval)
    timeout = max(5.0, args.timeout)
    status_path, request_path = load_paths()

    if not status_path.exists():
        print(f"상태 파일 없음: {status_path}")
        print("main_vp.py를 RUN 모드로 먼저 실행하세요.")
        return 2

    initial = read_json(status_path)
    if initial.get("state") != "RUNNING":
        print(f"카메라 상태가 RUNNING이 아님: {initial.get('state')}")
        return 2

    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_dir = PROJECT_ROOT / "data" / "logs" / "camera_recovery_stress" / stamp
    log_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = log_dir / "camera_recovery_stress.jsonl"
    summary_path = log_dir / "camera_recovery_stress_summary.json"

    results = []
    print(f"카메라 자동복구 스트레스 시작: {cycles}회")
    print(f"상태: {status_path}")

    for index in range(1, cycles + 1):
        before = read_json(status_path)
        before_count = int(before.get("recovery_count", 0))
        request_id = f"stress-{stamp}-{index:03d}"
        request = {
            "action": "simulate_hang",
            "request_id": request_id,
            "created_at": datetime.now().isoformat(timespec="milliseconds"),
            "cycle": index,
        }
        atomic_write_json(request_path, request)
        started = time.monotonic()
        print(f"[{index}/{cycles}] stall 요청: {request_id}")

        def completed():
            status = read_json(status_path)
            count = int(status.get("recovery_count", 0))
            if count <= before_count:
                return None
            state = str(status.get("state", ""))
            result = str(status.get("last_recovery_result", ""))
            if state == "RUNNING" and result == "OK":
                return status
            if state == "FAILED" or result == "FAIL":
                return status
            return None

        after = wait_until(completed, timeout)
        elapsed = time.monotonic() - started
        passed = bool(
            after
            and after.get("state") == "RUNNING"
            and after.get("last_recovery_result") == "OK"
        )
        record = {
            "timestamp": datetime.now().isoformat(timespec="milliseconds"),
            "cycle": index,
            "request_id": request_id,
            "pass": passed,
            "wall_elapsed_sec": round(elapsed, 3),
            "before": before,
            "after": after,
        }
        results.append(record)
        with jsonl_path.open("a", encoding="utf-8") as file:
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

        if not passed:
            print(f"[{index}/{cycles}] FAIL elapsed={elapsed:.2f}s")
            break

        recovery_elapsed = float(after.get("recovery_elapsed_sec", 0.0) or 0.0)
        print(
            f"[{index}/{cycles}] PASS recovery={recovery_elapsed:.3f}s "
            f"worker_generation={after.get('generation')}"
        )
        if index < cycles and interval > 0:
            time.sleep(interval)

    passed_count = sum(1 for item in results if item["pass"])
    summary = {
        "started_at": stamp,
        "requested_cycles": cycles,
        "completed_cycles": len(results),
        "passed_cycles": passed_count,
        "failed_cycles": len(results) - passed_count,
        "pass": passed_count == cycles,
        "average_recovery_sec": round(
            sum(
                float((item.get("after") or {}).get("recovery_elapsed_sec", 0.0) or 0.0)
                for item in results
                if item["pass"]
            )
            / max(1, passed_count),
            3,
        ),
        "log_path": str(jsonl_path),
        "status_path": str(status_path),
    }
    with summary_path.open("w", encoding="utf-8") as file:
        json.dump(summary, file, ensure_ascii=False, indent=2)

    print(json.dumps(summary, ensure_ascii=False, indent=2))
    return 0 if summary["pass"] else 1


if __name__ == "__main__":
    sys.exit(main())
