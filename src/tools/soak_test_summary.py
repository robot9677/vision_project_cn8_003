#!/usr/bin/env python3
import argparse
import json
import os
from typing import Dict, Optional


PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
DEFAULT_ROOT = os.path.join(PROJECT_ROOT, "data", "logs", "soak_tests")


def _latest_file(root: str, filename: str) -> Optional[str]:
    found = []
    if not os.path.isdir(root):
        return None
    for dir_path, _, filenames in os.walk(root):
        if filename not in filenames:
            continue
        path = os.path.join(dir_path, filename)
        try:
            found.append((os.path.getmtime(path), path))
        except OSError:
            pass
    found.sort(reverse=True)
    return found[0][1] if found else None


def _summary_from_jsonl(path: str) -> Dict:
    result = {
        "active_or_incomplete": True,
        "cycle_count": 0,
        "ok_count": 0,
        "ng_count": 0,
        "error_count": 0,
        "last_event": "",
        "last_timestamp": "",
    }
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            try:
                row = json.loads(line)
            except Exception:
                continue
            counts = row.get("counts") or {}
            result["cycle_count"] = int(row.get("cycle", result["cycle_count"]))
            result["ok_count"] = int(counts.get("ok", result["ok_count"]))
            result["ng_count"] = int(counts.get("ng", result["ng_count"]))
            result["error_count"] = int(counts.get("error", result["error_count"]))
            result["last_event"] = str(row.get("event", ""))
            result["last_timestamp"] = str(row.get("timestamp", ""))
            result["session_id"] = str(row.get("session_id", ""))
    return result


def main():
    parser = argparse.ArgumentParser(description="Show latest overnight soak test summary")
    parser.add_argument("--root", default=DEFAULT_ROOT)
    args = parser.parse_args()
    root = os.path.abspath(args.root)

    summary_path = _latest_file(root, "soak_summary.json")
    jsonl_path = _latest_file(root, "soak_cycle.jsonl")

    use_summary = False
    if summary_path and jsonl_path:
        use_summary = os.path.getmtime(summary_path) >= os.path.getmtime(jsonl_path)
    elif summary_path:
        use_summary = True

    if use_summary:
        with open(summary_path, "r", encoding="utf-8") as file:
            summary = json.load(file)
        source = summary_path
    elif jsonl_path:
        summary = _summary_from_jsonl(jsonl_path)
        source = jsonl_path
    else:
        print(f"No soak test logs found: {root}")
        return 1

    print("=" * 72)
    print("LATEST SOAK TEST")
    print("=" * 72)
    print(f"Source   : {source}")
    print(f"Session  : {summary.get('session_id', '-')}")
    print(f"Phase    : {summary.get('phase', summary.get('last_event', '-'))}")
    print(f"Started  : {summary.get('started_at', '-')}")
    print(f"Duration : {float(summary.get('duration_sec', 0.0)) / 3600.0:.2f} h")
    print(f"Cycles   : {summary.get('cycle_count', 0)}")
    print(f"OK       : {summary.get('ok_count', 0)}")
    print(f"NG       : {summary.get('ng_count', 0)}")
    print(f"Errors   : {summary.get('error_count', 0)}")
    print(f"Last     : {summary.get('last_result', summary.get('last_timestamp', '-'))}")
    print(f"Stop     : {summary.get('stop_reason', '-')}")
    print(f"Log      : {summary.get('log_path', jsonl_path or '-')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
