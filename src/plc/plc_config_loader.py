import copy
import json
import os
from typing import Any, Dict


DEFAULT_PLC_CONFIG: Dict[str, Any] = {
    "enabled": False,
    "backend": "modbus_rtu_slave",

    "serial": {
        "port": "/dev/ttyUSB0",
        "baudrate": 9600,
        "bytesize": 8,
        "parity": "N",
        "stopbits": 1,
        "timeout": 0.1,
    },

    "modbus": {
        "slave_id": 1,
    },

    "registers": {
        "command": 200,
        "status": 201,
        "result": 202,
        "heartbeat": 203,
        "error_code": 204,
        "last_inspect_time": 205,
        "ack": 206,
        "ready_detail": 207,
        "reserved1": 208,
        "reserved2": 209,
    },

    "heartbeat": {
        "interval_sec": 0.5,
        "max_value": 9999,
    },

    "shutdown": {
        "command": "sudo /sbin/shutdown -h now",
        "ready_hold_sec": 2.0,
    },
}


def _merge_default(base: Dict[str, Any], user: Dict[str, Any]) -> Dict[str, Any]:
    out = copy.deepcopy(base)

    for key, value in user.items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key].update(value)
        else:
            out[key] = value

    return out


def load_plc_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[PLC CONFIG] not found, use default: {path}")
        return copy.deepcopy(DEFAULT_PLC_CONFIG)

    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"plc_config must be object: {path}")

    return _merge_default(DEFAULT_PLC_CONFIG, cfg)