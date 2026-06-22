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
        "command": 0,
        "status": 1,
        "result": 2,
    },
}


def load_plc_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[PLC CONFIG] not found, use default: {path}")
        return DEFAULT_PLC_CONFIG.copy()

    with open(path, "r", encoding="utf-8-sig") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"plc_config must be object: {path}")

    return cfg