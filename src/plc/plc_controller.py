from typing import Any, Dict, Optional


class DisabledPlcController:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = False

    def start(self):
        print("[PLC] disabled")

    def stop(self):
        pass

    def poll_command(self) -> Optional[str]:
        return None

    def set_idle(self):
        pass

    def set_busy(self):
        pass

    def set_done(self, ok: bool):
        pass

    def set_error(self):
        pass


class PlcControllerStub:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))
        self.backend = str(cfg.get("backend", "modbus_rtu_slave"))

    def start(self):
        serial_cfg = self.cfg.get("serial", {}) or {}
        port = serial_cfg.get("port", "/dev/ttyUSB0")
        baudrate = serial_cfg.get("baudrate", 9600)
        print(f"[PLC] stub enabled backend={self.backend} port={port} baudrate={baudrate}")
        print("[PLC] real modbus handler is not attached yet")

    def stop(self):
        print("[PLC] stopped")

    def poll_command(self) -> Optional[str]:
        return None

    def set_idle(self):
        pass

    def set_busy(self):
        pass

    def set_done(self, ok: bool):
        pass

    def set_error(self):
        pass


def create_plc_controller(cfg: Dict[str, Any]):
    if not bool(cfg.get("enabled", False)):
        return DisabledPlcController(cfg)

    return PlcControllerStub(cfg)