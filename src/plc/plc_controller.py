import threading
import time
from typing import Any, Dict, Optional

try:
    import serial
except Exception:
    serial = None


def _crc16_modbus(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def _append_crc(data: bytes) -> bytes:
    crc = _crc16_modbus(data)
    return data + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def _check_crc(frame: bytes) -> bool:
    if len(frame) < 4:
        return False
    recv = frame[-2] | (frame[-1] << 8)
    calc = _crc16_modbus(frame[:-2])
    return recv == calc


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


class ModbusRtuSlaveController:
    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))

        serial_cfg = cfg.get("serial", {}) or {}
        modbus_cfg = cfg.get("modbus", {}) or {}
        regs_cfg = cfg.get("registers", {}) or {}

        self.port = str(serial_cfg.get("port", "/dev/ttyUSB0"))
        self.baudrate = int(serial_cfg.get("baudrate", 9600))
        self.bytesize = int(serial_cfg.get("bytesize", 8))
        self.parity = str(serial_cfg.get("parity", "N"))
        self.stopbits = int(serial_cfg.get("stopbits", 1))
        self.timeout = float(serial_cfg.get("timeout", 0.05))

        self.slave_id = int(modbus_cfg.get("slave_id", 1))

        self.reg_command = int(regs_cfg.get("command", 0))
        self.reg_status = int(regs_cfg.get("status", 1))
        self.reg_result = int(regs_cfg.get("result", 2))

        self.register_count = max(self.reg_command, self.reg_status, self.reg_result) + 16
        self.regs = [0] * self.register_count

        self._lock = threading.Lock()
        self._ser = None
        self._thread = None
        self._stop = threading.Event()

    def start(self):
        if serial is None:
            print("[PLC] pyserial is not installed - PLC disabled")
            self._ser = None
            return

        try:
            self._ser = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout,
            )
        except Exception as e:
            self._ser = None
            print(f"[PLC] port open failed - PLC disabled: port={self.port}, error={e}")
            return

        self.set_idle()

        self._stop.clear()
        self._thread = threading.Thread(target=self._loop, daemon=True)
        self._thread.start()

        print(
            f"[PLC] modbus rtu slave started "
            f"port={self.port} baudrate={self.baudrate} slave_id={self.slave_id}"
        )

    def stop(self):
        self._stop.set()

        if self._thread is not None:
            self._thread.join(timeout=1.0)
            self._thread = None

        if self._ser is not None:
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

        print("[PLC] stopped")

    def poll_command(self) -> Optional[str]:
        cmd = self._get_reg(self.reg_command)

        if cmd == 1:
            self._set_reg(self.reg_command, 0)
            print("[PLC] command received: PREPARE_REQUEST")
            return "prepare"

        if cmd == 2:
            self._set_reg(self.reg_command, 0)
            print("[PLC] command received: INSPECT_START")
            return "inspect"

        return None

    def set_idle(self):
        self._set_reg(self.reg_status, 0)
        self._set_reg(self.reg_result, 0)

    def set_busy(self):
        self._set_reg(self.reg_status, 1)
        self._set_reg(self.reg_result, 0)

    def set_done(self, ok: bool):
        self._set_reg(self.reg_status, 2)
        self._set_reg(self.reg_result, 1 if ok else 2)
        print(f"[PLC] result set: {'OK' if ok else 'NG'}")

    def set_error(self):
        self._set_reg(self.reg_status, 3)
        self._set_reg(self.reg_result, 0)
        print("[PLC] result set: ERROR")

    def _get_reg(self, idx: int) -> int:
        with self._lock:
            if 0 <= idx < len(self.regs):
                return int(self.regs[idx])
            return 0

    def _set_reg(self, idx: int, val: int):
        with self._lock:
            if 0 <= idx < len(self.regs):
                self.regs[idx] = int(val) & 0xFFFF

    def _read_exact(self, n: int) -> Optional[bytes]:
        if self._ser is None:
            return None

        data = self._ser.read(n)
        if len(data) != n:
            return None
        return data

    def _loop(self):
        while not self._stop.is_set():
            try:
                b1 = self._read_exact(1)
                if not b1:
                    continue

                b2 = self._read_exact(1)
                if not b2:
                    continue

                addr = b1[0]
                func = b2[0]

                if func in (3, 4, 6):
                    rest = self._read_exact(6)
                    if not rest:
                        continue
                    frame = b1 + b2 + rest

                elif func == 16:
                    head = self._read_exact(5)
                    if not head:
                        continue
                    byte_count = int(head[4])
                    tail = self._read_exact(byte_count + 2)
                    if not tail:
                        continue
                    frame = b1 + b2 + head + tail

                else:
                    continue

                if not _check_crc(frame):
                    continue

                if addr not in (self.slave_id, 0):
                    continue

                if func in (3, 4):
                    self._handle_read_holding(addr, frame)

                elif func == 6:
                    self._handle_write_single(addr, frame)

                elif func == 16:
                    self._handle_write_multiple(addr, frame)

            except Exception as e:
                print("[PLC] loop error:", e)
                time.sleep(0.1)

    def _write_response(self, addr: int, payload: bytes):
        if addr == 0:
            return
        if self._ser is None:
            return

        try:
            self._ser.write(_append_crc(payload))
        except Exception as e:
            print("[PLC] write failed:", e)
            try:
                self._ser.close()
            except Exception:
                pass
            self._ser = None

    def _write_exception(self, addr: int, func: int, code: int):
        if addr == 0:
            return
        self._write_response(addr, bytes([addr, func | 0x80, code]))

    def _handle_read_holding(self, addr: int, frame: bytes):
        func = frame[1]
        start = (frame[2] << 8) | frame[3]
        qty = (frame[4] << 8) | frame[5]

        if qty <= 0 or qty > 32:
            self._write_exception(addr, func, 3)
            return

        if start < 0 or start + qty > len(self.regs):
            self._write_exception(addr, func, 2)
            return

        with self._lock:
            vals = self.regs[start:start + qty]

        data = bytearray()
        for v in vals:
            data.append((int(v) >> 8) & 0xFF)
            data.append(int(v) & 0xFF)

        payload = bytes([addr, func, len(data)]) + bytes(data)
        self._write_response(addr, payload)

    def _handle_write_single(self, addr: int, frame: bytes):
        func = frame[1]
        reg = (frame[2] << 8) | frame[3]
        val = (frame[4] << 8) | frame[5]

        if reg < 0 or reg >= len(self.regs):
            self._write_exception(addr, func, 2)
            return

        self._set_reg(reg, val)

        if addr != 0:
            self._write_response(addr, frame[:-2])

    def _handle_write_multiple(self, addr: int, frame: bytes):
        func = frame[1]
        start = (frame[2] << 8) | frame[3]
        qty = (frame[4] << 8) | frame[5]
        byte_count = frame[6]

        if qty <= 0 or qty > 32 or byte_count != qty * 2:
            self._write_exception(addr, func, 3)
            return

        if start < 0 or start + qty > len(self.regs):
            self._write_exception(addr, func, 2)
            return

        pos = 7
        for i in range(qty):
            val = (frame[pos] << 8) | frame[pos + 1]
            self._set_reg(start + i, val)
            pos += 2

        payload = bytes([
            addr,
            func,
            (start >> 8) & 0xFF,
            start & 0xFF,
            (qty >> 8) & 0xFF,
            qty & 0xFF,
        ])
        self._write_response(addr, payload)


def create_plc_controller(cfg: Dict[str, Any]):
    if not bool(cfg.get("enabled", False)):
        return DisabledPlcController(cfg)

    backend = str(cfg.get("backend", "modbus_rtu_slave")).strip().lower()

    if backend == "modbus_rtu_slave":
        return ModbusRtuSlaveController(cfg)

    raise ValueError(f"unsupported plc backend: {backend}")