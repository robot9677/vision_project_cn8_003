#!/usr/bin/env python3
"""Print the latest centralized diagnostic index entries."""

import argparse
import json
import os
from collections import deque


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DEFAULT_INDEX = os.path.join(
    PROJECT_ROOT,
    "data",
    "logs",
    "diagnostics",
    "diagnostics_index.jsonl",
)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=20)
    parser.add_argument("--path", default=DEFAULT_INDEX)
    args = parser.parse_args()

    path = os.path.abspath(args.path)
    if not os.path.exists(path):
        print(f"Diagnostic index not found: {path}")
        return 1

    rows = deque(maxlen=max(1, int(args.limit)))
    with open(path, "r", encoding="utf-8", errors="replace") as file:
        for line in file:
            line = line.strip()
            if not line:
                continue
            try:
                rows.append(json.loads(line))
            except json.JSONDecodeError:
                continue

    print(f"DIAGNOSTIC INDEX: {path}")
    for row in rows:
        print(
            f"{row.get('timestamp', '')} "
            f"{str(row.get('kind', '')).upper():16s} "
            f"E{int(row.get('error_code', 0)):02d} "
            f"{row.get('phase', row.get('event_type', ''))} "
            f"{row.get('result', '')} "
            f"{row.get('path', '')}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
