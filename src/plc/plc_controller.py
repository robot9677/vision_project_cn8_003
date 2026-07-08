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

    def tick(self):
        pass

    def poll_command(self) -> Optional[str]:
        return None

    def poll_ack(self) -> Optional[str]:
        return None

    def set_idle(self):
        pass

    def set_busy(self):
        pass

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        pass

    def set_error(self, code: int = 99):
        pass

    def set_shutdown_busy(self):
        pass

    def set_shutdown_ready(self):
        pass

    def set_ready_detail(self, value: int):
        pass


class ModbusRtuSlaveController:
    CMD_NONE = 0
    CMD_PREPARE = 1
    CMD_INSPECT = 2
    CMD_SHUTDOWN = 8
    CMD_EMERGENCY = 9

    ACK_NONE = 0
    ACK_RESULT = 1
    ACK_ERROR_RESET = 2

    STATUS_READY = 0
    STATUS_BUSY = 1
    STATUS_DONE = 2
    STATUS_ERROR = 3
    STATUS_SHUTDOWN_BUSY = 8
    STATUS_SHUTDOWN_READY = 9

    RESULT_NONE = 0
    RESULT_OK = 1
    RESULT_NG = 2
    RESULT_FAIL = 3

    def __init__(self, cfg: Dict[str, Any]):
        self.cfg = cfg
        self.enabled = bool(cfg.get("enabled", False))

        serial_cfg = cfg.get("serial", {}) or {}
        modbus_cfg = cfg.get("modbus", {}) or {}
        regs_cfg = cfg.get("registers", {}) or {}
        heartbeat_cfg = cfg.get("heartbeat", {}) or {}

        self.port = str(serial_cfg.get("port", "/dev/ttyUSB0"))
        self.baudrate = int(serial_cfg.get("baudrate", 9600))
        self.bytesize = int(serial_cfg.get("bytesize", 8))
        self.parity = str(serial_cfg.get("parity", "N"))
        self.stopbits = int(serial_cfg.get("stopbits", 1))
        self.timeout = float(serial_cfg.get("timeout", 0.05))

        self.slave_id = int(modbus_cfg.get("slave_id", 1))

        self.reg_command = int(regs_cfg.get("command", 200))
        self.reg_status = int(regs_cfg.get("status", 201))
        self.reg_result = int(regs_cfg.get("result", 202))
        self.reg_heartbeat = int(regs_cfg.get("heartbeat", 203))
        self.reg_error_code = int(regs_cfg.get("error_code", 204))
        self.reg_last_inspect_time = int(regs_cfg.get("last_inspect_time", 205))
        self.reg_ack = int(regs_cfg.get("ack", 206))
        self.reg_ready_detail = int(regs_cfg.get("ready_detail", 207))
        self.reg_reserved1 = int(regs_cfg.get("reserved1", 208))
        self.reg_reserved2 = int(regs_cfg.get("reserved2", 209))

        max_reg = max(
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
            self.reg_error_code,
            self.reg_last_inspect_time,
            self.reg_ack,
            self.reg_ready_detail,
            self.reg_reserved1,
            self.reg_reserved2,
        )

        self.register_count = max_reg + 16
        self.regs = [0] * self.register_count

        self.heartbeat_interval_sec = float(heartbeat_cfg.get("interval_sec", 0.5))
        self.heartbeat_max = int(heartbeat_cfg.get("max_value", 9999))
        self._last_heartbeat_ts = 0.0

        self._last_cmd_seen = 0
        self._last_ack_seen = 0

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

    def tick(self):
        now = time.time()
        if (now - self._last_heartbeat_ts) < self.heartbeat_interval_sec:
            return

        self._last_heartbeat_ts = now

        current = self._get_reg(self.reg_heartbeat)
        next_val = current + 1

        if next_val > self.heartbeat_max:
            next_val = 0

        self._set_reg(self.reg_heartbeat, next_val)

    def poll_command(self) -> Optional[str]:
        cmd = self._get_reg(self.reg_command)

        if cmd == self.CMD_NONE:
            self._last_cmd_seen = self.CMD_NONE
            return None

        if cmd == self._last_cmd_seen:
            return None

        self._last_cmd_seen = cmd

        if cmd == self.CMD_PREPARE:
            print("[PLC] command received: PREPARE_REQUEST")
            return "prepare"

        if cmd == self.CMD_INSPECT:
            print("[PLC] command received: INSPECT_START")
            return "inspect"

        if cmd == self.CMD_SHUTDOWN:
            print("[PLC] command received: SHUTDOWN_REQUEST")
            return "shutdown"

        if cmd == self.CMD_EMERGENCY:
            print("[PLC] command received: EMERGENCY_STOP")
            return "emergency"

        print(f"[PLC] unknown command: {cmd}")
        return "command_error"

    def poll_ack(self) -> Optional[str]:
        ack = self._get_reg(self.reg_ack)

        if ack == self.ACK_NONE:
            self._last_ack_seen = self.ACK_NONE
            return None

        if ack == self._last_ack_seen:
            return None

        self._last_ack_seen = ack

        if ack == self.ACK_RESULT:
            print("[PLC] ack received: RESULT_ACK")
            return "result_ack"

        if ack == self.ACK_ERROR_RESET:
            print("[PLC] ack received: ERROR_RESET")
            return "error_reset"

        print(f"[PLC] unknown ack: {ack}")
        return "ack_error"

    def set_idle(self):
        self._set_reg(self.reg_status, self.STATUS_READY)
        self._set_reg(self.reg_result, self.RESULT_NONE)
        self._set_reg(self.reg_error_code, 0)
        self._set_reg(self.reg_last_inspect_time, 0)
        self._set_reg(self.reg_ready_detail, 0)

    def set_busy(self):
        self._set_reg(self.reg_status, self.STATUS_BUSY)
        self._set_reg(self.reg_result, self.RESULT_NONE)
        self._set_reg(self.reg_error_code, 0)

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        self._set_reg(self.reg_status, self.STATUS_DONE)
        self._set_reg(self.reg_result, self.RESULT_OK if ok else self.RESULT_NG)
        self._set_reg(self.reg_error_code, 0)
        self._set_reg(self.reg_last_inspect_time, max(0, int(elapsed_ms)))

        print(f"[PLC] result set: {'OK' if ok else 'NG'} elapsed_ms={int(elapsed_ms)}")

    def set_error(self, code: int = 99):
        self._set_reg(self.reg_status, self.STATUS_ERROR)
        self._set_reg(self.reg_result, self.RESULT_FAIL)
        self._set_reg(self.reg_error_code, int(code))

        print(f"[PLC] result set: ERROR code={int(code)}")

    def set_shutdown_busy(self):
        self._set_reg(self.reg_status, self.STATUS_SHUTDOWN_BUSY)
        self._set_reg(self.reg_result, self.RESULT_NONE)
        self._set_reg(self.reg_error_code, 0)

        print("[PLC] shutdown busy")

    def set_shutdown_ready(self):
        self._set_reg(self.reg_status, self.STATUS_SHUTDOWN_READY)
        self._set_reg(self.reg_result, self.RESULT_NONE)
        self._set_reg(self.reg_error_code, 0)

        print("[PLC] shutdown ready")

    def set_ready_detail(self, value: int):
        self._set_reg(self.reg_ready_detail, int(value))

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