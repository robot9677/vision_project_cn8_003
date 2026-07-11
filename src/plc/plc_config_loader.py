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
        "reconnect_interval_sec": 1.0,
    },

    "modbus": {
        "slave_id": 1,
    },

    "registers": {
        "command": 200,
        "status": 201,
        "result": 202,
        "heartbeat": 203,
    },

    "heartbeat": {
        "interval_sec": 0.5,
        "max_value": 9999,
    },

    "recovery": {
        "camera_open_timeout_sec": 2.0,
        "camera_retry_interval_sec": 0.1,
        "camera_grace_sec": 2.0,
        "frame_timeout_sec": 1.0,
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