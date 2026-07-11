#!/usr/bin/env python3
"""PLC Modbus RTU live trace monitor.

This program does not open the serial port. It follows the JSONL trace written
by src/plc/plc_controller.py, so it can run while main_vp.py owns /dev/ttyUSB*.
"""

import argparse
import json
import os
import sys
import time
from collections import deque
from typing import Any, Deque, Dict, Optional


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
DEFAULT_TRACE_PATH = os.path.join(
    PROJECT_ROOT,
    "data",
    "logs",
    "plc_live.jsonl",
)

CMD_NAMES = {
    0: "NONE",
    1: "PREPARE",
    2: "INSPECT",
    3: "RESET",
    8: "SHUTDOWN",
    9: "EMERGENCY",
}

STATUS_NAMES = {
    0: "READY",
    1: "BUSY",
    2: "DONE",
    3: "ERROR",
    8: "SHUTDOWN",
}

RESULT_NAMES = {
    0: "NONE",
    1: "OK",
    2: "NG",
}


def _name(mapping: Dict[int, str], value: Any) -> str:
    try:
        number = int(value)
    except Exception:
        return "UNKNOWN"
    return mapping.get(number, f"UNKNOWN({number})")


def _age_text(epoch: Optional[float], now: float) -> str:
    if epoch is None:
        return "NONE"

    age = max(0.0, now - float(epoch))
    if age < 10.0:
        return f"{age:.1f}s"
    if age < 60.0:
        return f"{age:.0f}s"
    return f"{age / 60.0:.1f}m"


def _short_hex(value: str, limit: int = 86) -> str:
    text = str(value or "")
    if len(text) <= limit:
        return text
    return text[: limit - 3] + "..."


def _format_event(event: Dict[str, Any]) -> str:
    timestamp = str(event.get("timestamp", ""))
    clock = timestamp[11:23] if len(timestamp) >= 23 else timestamp
    direction = str(event.get("direction", "?"))
    summary = str(event.get("summary", ""))
    raw_hex = _short_hex(str(event.get("hex", "")))

    if raw_hex:
        return f"{clock} {direction:6s} {summary} | {raw_hex}"
    return f"{clock} {direction:6s} {summary}"


class TraceFollower:
    def __init__(self, path: str):
        self.path = os.path.abspath(path)
        self._fp = None
        self._inode = None
        self._offset = 0

    def close(self):
        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
        self._fp = None
        self._inode = None
        self._offset = 0

    def _open(self):
        self.close()
        self._fp = open(self.path, "r", encoding="utf-8", errors="replace")
        stat = os.fstat(self._fp.fileno())
        self._inode = (stat.st_dev, stat.st_ino)
        self._offset = 0

    def read_new(self):
        if not os.path.exists(self.path):
            self.close()
            return []

        try:
            stat = os.stat(self.path)
            inode = (stat.st_dev, stat.st_ino)

            if self._fp is None or inode != self._inode:
                self._open()

            if stat.st_size < self._offset:
                self._open()

            self._fp.seek(self._offset)
            lines = self._fp.readlines()
            self._offset = self._fp.tell()

        except (FileNotFoundError, OSError):
            self.close()
            return []

        events = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError:
                continue
            if isinstance(event, dict):
                events.append(event)

        return events


def _clear_screen(enabled: bool):
    if enabled:
        sys.stdout.write("\033[2J\033[H")


def run_stream(follower: TraceFollower, interval: float):
    print(f"[PLC LIVE STREAM] {follower.path}")
    print("Ctrl+C to stop. The vision program continues running.\n")

    while True:
        events = follower.read_new()
        for event in events:
            print(_format_event(event), flush=True)
        time.sleep(interval)


def run_dashboard(
    follower: TraceFollower,
    interval: float,
    history_size: int,
    stale_sec: float,
    clear_screen: bool,
):
    history: Deque[Dict[str, Any]] = deque(maxlen=history_size)
    registers = {"D200": 0, "D201": 0, "D202": 0, "D203": 0}

    serial_open = False
    comm_fault = False
    last_rx_epoch = None
    last_tx_epoch = None
    last_event_epoch = None
    last_heartbeat_value = None
    last_heartbeat_change_epoch = None
    parse_count = 0

    while True:
        events = follower.read_new()
        for event in events:
            parse_count += 1
            last_event_epoch = event.get("epoch", last_event_epoch)

            event_registers = event.get("registers")
            if isinstance(event_registers, dict):
                for key in registers:
                    if key in event_registers:
                        registers[key] = event_registers[key]

            serial_open = bool(event.get("serial_open", serial_open))
            comm_fault = bool(event.get("comm_fault_active", comm_fault))

            direction = str(event.get("direction", ""))
            epoch = event.get("epoch")
            if direction == "RX":
                last_rx_epoch = epoch
                history.append(event)
            elif direction == "TX":
                last_tx_epoch = epoch
                history.append(event)
            elif direction == "SYSTEM":
                history.append(event)
            elif direction == "STATE":
                register = event.get("register")
                if register == 203:
                    heartbeat_value = event.get("new_value")
                    if heartbeat_value != last_heartbeat_value:
                        last_heartbeat_value = heartbeat_value
                        last_heartbeat_change_epoch = epoch
                elif register in (200, 201, 202):
                    history.append(event)

        now = time.time()
        heartbeat_age = (
            None
            if last_heartbeat_change_epoch is None
            else now - float(last_heartbeat_change_epoch)
        )
        heartbeat_moving = (
            heartbeat_age is not None
            and heartbeat_age <= max(1.5, stale_sec)
        )
        plc_polling = (
            last_rx_epoch is not None
            and now - float(last_rx_epoch) <= stale_sec
        )

        _clear_screen(clear_screen)

        print("=" * 94)
        print(" PLC LIVE MONITOR  |  main_vp.py 실행 상태에서 사용  |  Ctrl+C: 모니터만 종료")
        print("=" * 94)
        print(f" Trace file     : {follower.path}")
        print(
            f" Vision serial  : {'OPEN' if serial_open else 'CLOSED'}"
            f"    COMM fault: {'YES' if comm_fault else 'NO'}"
            f"    Parsed events: {parse_count}"
        )
        print(
            f" PLC polling RX : {'ACTIVE' if plc_polling else 'STALE/NONE'}"
            f"    last RX: {_age_text(last_rx_epoch, now)}"
            f"    last TX: {_age_text(last_tx_epoch, now)}"
            f"    last event: {_age_text(last_event_epoch, now)}"
        )
        print(
            f" Heartbeat      : {'MOVING' if heartbeat_moving else 'STOPPED/UNKNOWN'}"
            f"    D203={registers['D203']}"
            f"    last change: {_age_text(last_heartbeat_change_epoch, now)}"
        )
        print("-" * 94)
        print(
            f" D200 COMMAND : {int(registers['D200']):5d}  "
            f"{_name(CMD_NAMES, registers['D200'])}"
        )
        print(
            f" D201 STATUS  : {int(registers['D201']):5d}  "
            f"{_name(STATUS_NAMES, registers['D201'])}"
        )
        print(
            f" D202 RESULT  : {int(registers['D202']):5d}  "
            f"{_name(RESULT_NAMES, registers['D202'])}"
        )
        print(f" D203 HEART   : {int(registers['D203']):5d}")
        print("-" * 94)
        print(" Recent PLC RX / Vision TX / state events")
        print("-" * 94)

        if history:
            for event in history:
                print(_format_event(event))
        else:
            if not os.path.exists(follower.path):
                print(" Trace file waiting... Restart main_vp.py after installing the modified files.")
            else:
                print(" Trace file exists, but no events have been received yet.")

        print("=" * 94, flush=True)
        time.sleep(interval)


def main():
    parser = argparse.ArgumentParser(
        description=(
            "Monitor PLC RX/TX and D200-D203 without opening the RS485 port."
        )
    )
    parser.add_argument(
        "--file",
        default=DEFAULT_TRACE_PATH,
        help=f"trace JSONL path (default: {DEFAULT_TRACE_PATH})",
    )
    parser.add_argument(
        "--interval",
        type=float,
        default=0.2,
        help="screen refresh interval seconds (default: 0.2)",
    )
    parser.add_argument(
        "--history",
        type=int,
        default=12,
        help="recent event lines shown in dashboard (default: 12)",
    )
    parser.add_argument(
        "--stale-sec",
        type=float,
        default=2.0,
        help="RX/heartbeat stale threshold seconds (default: 2.0)",
    )
    parser.add_argument(
        "--stream",
        action="store_true",
        help="print every event instead of the dashboard",
    )
    parser.add_argument(
        "--no-clear",
        action="store_true",
        help="do not clear the terminal between dashboard refreshes",
    )
    args = parser.parse_args()

    follower = TraceFollower(args.file)

    try:
        if args.stream:
            run_stream(follower, max(0.02, args.interval))
        else:
            run_dashboard(
                follower=follower,
                interval=max(0.05, args.interval),
                history_size=max(1, args.history),
                stale_sec=max(0.5, args.stale_sec),
                clear_screen=not args.no_clear and sys.stdout.isatty(),
            )
    except KeyboardInterrupt:
        print("\n[PLC LIVE MONITOR] stopped. Vision continues running.")
    finally:
        follower.close()


if __name__ == "__main__":
    main()
