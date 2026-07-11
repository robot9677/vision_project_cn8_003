import threading
import time
from typing import Any, Dict, Optional
from datetime import datetime

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

    def set_idle(self):
        pass

    def reset_to_ready(self, reset_heartbeat: bool = False):
        pass

    def set_busy(self):
        pass

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        pass

    def set_error(self, code: int = 99, detail: str = ""):
        pass

    def set_shutdown(self):
        pass

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "command": 0,
            "status": 0,
            "result": 0,
            "heartbeat": 0,
            "error_code": 0,
            "error_detail": "",
            "last_inspect_at": "",
            "last_inspect_elapsed_ms": 0,
            "serial_open": False,
            "comm_fault_active": False,
        }

    def is_connected(self) -> bool:
        return False
    
    def poll_comm_event(self) -> Optional[Dict[str, Any]]:
        return None

    def prepare_reset(self, reset_heartbeat: bool = False):
        pass


class ModbusRtuSlaveController:
    CMD_NONE = 0
    CMD_PREPARE = 1
    CMD_INSPECT = 2
    CMD_RESET = 3
    CMD_SHUTDOWN = 8
    CMD_EMERGENCY = 9

    STATUS_READY = 0
    STATUS_BUSY = 1
    STATUS_DONE = 2
    STATUS_ERROR = 3
    STATUS_SHUTDOWN = 8

    RESULT_NONE = 0
    RESULT_OK = 1
    RESULT_NG = 2

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
        self.timeout = max(0.01, float(serial_cfg.get("timeout", 0.05)))

        self.reconnect_interval_sec = max(
            0.1,
            float(serial_cfg.get("reconnect_interval_sec", 1.0)),
        )

        self.slave_id = int(modbus_cfg.get("slave_id", 1))

        self.reg_command = int(regs_cfg.get("command", 200))
        self.reg_status = int(regs_cfg.get("status", 201))
        self.reg_result = int(regs_cfg.get("result", 202))
        self.reg_heartbeat = int(regs_cfg.get("heartbeat", 203))

        if not 1 <= self.slave_id <= 247:
            raise ValueError(
                f"modbus slave_id must be 1..247: {self.slave_id}"
            )

        register_addresses = [
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
        ]

        if any(reg < 0 or reg > 0xFFFF for reg in register_addresses):
            raise ValueError(
                f"modbus register address out of range: {register_addresses}"
            )

        if len(set(register_addresses)) != len(register_addresses):
            raise ValueError(
                f"modbus register addresses must be unique: {register_addresses}"
            )

        max_reg = max(
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
        )

        self.register_count = max_reg + 1
        self.regs = [0] * self.register_count

        self.last_error_code = 0
        self.last_error_detail = ""
        self.last_inspect_at = ""
        self.last_inspect_elapsed_ms = 0

        self.heartbeat_interval_sec = max(
            0.1,
            float(heartbeat_cfg.get("interval_sec", 0.5)),
        )
        self.heartbeat_max = max(1, int(heartbeat_cfg.get("max_value", 9999)))
        self._last_heartbeat_ts = 0.0

        self._last_cmd_seen = 0

        self._lock = threading.Lock()
        self._ser = None
        self._thread = None
        self._heartbeat_thread = None
        self._stop = threading.Event()

        self._comm_event_lock = threading.Lock()
        self._comm_events = []
        self._comm_fault_active = False
        self._last_reconnect_ts = 0.0

    def start(self):
        if self._thread is not None and self._thread.is_alive():
            print("[PLC] already started")
            return

        self.reset_to_ready(reset_heartbeat=True)

        self._stop.clear()
        self._last_reconnect_ts = 0.0

        with self._comm_event_lock:
            self._comm_events.clear()

        connected = self._open_serial(initial=True)

        self._thread = threading.Thread(
            target=self._loop,
            daemon=True,
        )
        self._thread.start()

        self._heartbeat_thread = threading.Thread(
            target=self._heartbeat_loop,
            daemon=True,
        )
        self._heartbeat_thread.start()

        if connected:
            print(
                f"[PLC] modbus rtu slave started "
                f"port={self.port} "
                f"baudrate={self.baudrate} "
                f"slave_id={self.slave_id}"
            )
        else:
            print(
                f"[PLC] serial unavailable, reconnecting every "
                f"{self.reconnect_interval_sec:.1f} sec"
            )

    def _push_comm_event(
        self,
        error_code: int,
        message: str,
    ):
        event = {
            "error_code": int(error_code),
            "message": str(message),
            "timestamp": datetime.now().strftime(
                "%Y-%m-%d %H:%M:%S.%f"
            )[:-3],
        }

        with self._comm_event_lock:
            self._comm_events.append(event)

            if len(self._comm_events) > 20:
                self._comm_events.pop(0)

    def poll_comm_event(self) -> Optional[Dict[str, Any]]:
        with self._comm_event_lock:
            if not self._comm_events:
                return None

            return self._comm_events.pop(0)

    def _close_serial(self):
        ser = self._ser
        self._ser = None

        if ser is not None:
            try:
                ser.close()
            except Exception:
                pass

    def _mark_comm_error(
        self,
        error_code: int,
        message: str,
    ):
        self._close_serial()

        # 정상 종료 중 발생한 close/read 예외는 통신 장애로 기록하지 않는다.
        if self._stop.is_set():
            return

        if not self._comm_fault_active:
            self._push_comm_event(
                error_code=error_code,
                message=message,
            )

            self._comm_fault_active = True

            # 통신 복구 후 PLC가 읽을 수 있도록 내부 D201=3 유지
            self.set_error(
                code=error_code,
                detail=message,
            )

        print(f"[PLC] communication error: {message}")

    def _open_serial(self, initial: bool = False) -> bool:
        if self._stop.is_set():
            return False

        if serial is None:
            self._mark_comm_error(
                error_code=70,
                message="pyserial is not installed",
            )
            return False

        try:
            opened_serial = serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=self.bytesize,
                parity=self.parity,
                stopbits=self.stopbits,
                timeout=self.timeout,
            )

            if hasattr(opened_serial, "is_open") and not opened_serial.is_open:
                raise RuntimeError("serial port object is not open")

            try:
                opened_serial.reset_input_buffer()
                opened_serial.reset_output_buffer()
            except Exception:
                pass

            self._ser = opened_serial

            was_fault = self._comm_fault_active
            self._comm_fault_active = False

            if was_fault:
                print(
                    f"[PLC] serial reconnected: {self.port} "
                    f"- D201 remains ERROR until D200=3"
                )

            return True

        except Exception as e:
            error_code = 70 if initial else 71

            self._mark_comm_error(
                error_code=error_code,
                message=(
                    f"serial port open failed: "
                    f"port={self.port}, error={e}"
                ),
            )
            return False
        
    def stop(self):
        self._stop.set()

        # read() 대기를 즉시 해제한 뒤 스레드를 종료한다.
        self._close_serial()

        join_timeout = max(1.0, self.timeout + 0.5)

        if self._thread is not None:
            self._thread.join(timeout=join_timeout)
            self._thread = None

        if self._heartbeat_thread is not None:
            self._heartbeat_thread.join(timeout=join_timeout)
            self._heartbeat_thread = None

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

    def _heartbeat_loop(self):
        while not self._stop.wait(
            self.heartbeat_interval_sec
        ):
            self.tick()

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
            print("[PLC] command received: INSPECT_REQUEST")
            return "inspect"
        
        if cmd == self.CMD_RESET:
            print("[PLC] command received: RESET_REQUEST")
            return "reset"

        if cmd == self.CMD_SHUTDOWN:
            print("[PLC] command received: SHUTDOWN_REQUEST")
            return "shutdown"

        if cmd == self.CMD_EMERGENCY:
            print("[PLC] command received: EMERGENCY_STOP")
            return "emergency"

        print(f"[PLC] unknown command: {cmd}")
        return "command_error"

    def set_idle(self):
        # D202 결과는 D200=3이 들어오기 전까지 유지
        self._set_reg(self.reg_status, self.STATUS_READY)

    def prepare_reset(
        self,
        reset_heartbeat: bool = False,
    ):
        # D200은 PLC가 직접 0으로 복귀
        # D202는 Reset 명령 즉시 초기화
        self._set_reg(
            self.reg_result,
            self.RESULT_NONE,
        )

        if reset_heartbeat:
            self._set_reg(self.reg_heartbeat, 0)
            self._last_heartbeat_ts = time.time()

        print(
            f"[PLC] reset preparation "
            f"result_cleared=True "
            f"heartbeat_reset={bool(reset_heartbeat)}"
        )

    def reset_to_ready(self, reset_heartbeat: bool = False):
        self._set_reg(self.reg_status, self.STATUS_READY)
        self._set_reg(self.reg_result, self.RESULT_NONE)

        if reset_heartbeat:
            self._set_reg(self.reg_heartbeat, 0)
            self._last_heartbeat_ts = time.time()

        self.last_error_code = 0
        self.last_error_detail = ""

        self.last_inspect_at = ""
        self.last_inspect_elapsed_ms = 0

        print(
            f"[PLC] reset to ready "
            f"heartbeat_reset={bool(reset_heartbeat)}"
        )

    def set_busy(self):
        # 이전 D202 결과는 Reset 명령 전까지 유지
        self._set_reg(self.reg_status, self.STATUS_BUSY)

    def set_done(self, ok: bool, elapsed_ms: int = 0):
        self._set_reg(self.reg_status, self.STATUS_DONE)
        self._set_reg(
            self.reg_result,
            self.RESULT_OK if ok else self.RESULT_NG,
        )

        self.last_error_code = 0
        self.last_error_detail = ""
        self.last_inspect_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        self.last_inspect_elapsed_ms = max(0, int(elapsed_ms))

        print(
            f"[PLC] result set: {'OK' if ok else 'NG'} "
            f"inspect_at={self.last_inspect_at} "
            f"elapsed_ms={self.last_inspect_elapsed_ms}"
        )

    def set_error(self, code: int = 99, detail: str = ""):
        # D202의 기존 OK/NG 값은 유지하고 D201만 Error로 변경
        self._set_reg(self.reg_status, self.STATUS_ERROR)

        self.last_error_code = int(code)
        self.last_error_detail = str(detail or "")

        print(
            f"[PLC] error status "
            f"code={self.last_error_code} "
            f"detail={self.last_error_detail}"
        )

    def set_shutdown(self):
        # 종료 상태는 D201=8 하나만 사용
        self._set_reg(self.reg_status, self.STATUS_SHUTDOWN)
        print("[PLC] shutdown status set: D201=8")

    def get_state_snapshot(self) -> Dict[str, Any]:
        return {
            "command": self._get_reg(self.reg_command),
            "status": self._get_reg(self.reg_status),
            "result": self._get_reg(self.reg_result),
            "heartbeat": self._get_reg(self.reg_heartbeat),
            "error_code": int(self.last_error_code),
            "error_detail": str(self.last_error_detail),
            "last_inspect_at": str(self.last_inspect_at),
            "last_inspect_elapsed_ms": int(self.last_inspect_elapsed_ms),
            "serial_open": self.is_connected(),
            "comm_fault_active": bool(self._comm_fault_active),
        }

    def is_connected(self) -> bool:
        ser = self._ser
        if ser is None:
            return False

        try:
            return bool(getattr(ser, "is_open", True))
        except Exception:
            return False

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
        if n <= 0:
            return b""

        data = bytearray()
        deadline = time.monotonic() + self.timeout

        while len(data) < n and not self._stop.is_set():
            ser = self._ser
            if ser is None:
                return None

            try:
                chunk = ser.read(n - len(data))
            except Exception as e:
                self._mark_comm_error(
                    error_code=71,
                    message=f"serial read failed: {e}",
                )
                return None

            if chunk:
                data.extend(chunk)
                continue

            if time.monotonic() >= deadline:
                break

        if len(data) != n:
            return None

        return bytes(data)

    def _loop(self):
        while not self._stop.is_set():

            if self._ser is None:
                now = time.time()

                if (
                    now - self._last_reconnect_ts
                    >= self.reconnect_interval_sec
                ):
                    self._last_reconnect_ts = now
                    self._open_serial(initial=False)

                time.sleep(0.05)
                continue

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
                    # 지원하지 않는 function code의 나머지 바이트가
                    # 다음 프레임으로 섞이지 않도록 입력 버퍼를 비운다.
                    try:
                        if self._ser is not None:
                            self._ser.reset_input_buffer()
                    except Exception:
                        pass
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

        ser = self._ser
        if ser is None:
            return

        frame = _append_crc(payload)

        try:
            written = ser.write(frame)
            if written is not None and int(written) != len(frame):
                raise IOError(
                    f"short serial write: {written}/{len(frame)} bytes"
                )
            ser.flush()
        except Exception as e:
            self._mark_comm_error(
                error_code=71,
                message=f"serial write failed: {e}",
            )

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

        allowed_registers = {
            self.reg_command,
            self.reg_status,
            self.reg_result,
            self.reg_heartbeat,
        }

        requested_registers = range(
            start,
            start + qty,
        )

        if any(
            reg not in allowed_registers
            for reg in requested_registers
        ):
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

        if reg != self.reg_command:
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

        if start != self.reg_command or qty != 1:
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