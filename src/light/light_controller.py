import time

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


def _build_write_single_register(slave_id: int, register: int, value: int) -> bytes:
    frame = bytes([
        int(slave_id) & 0xFF,
        0x06,
        (int(register) >> 8) & 0xFF,
        int(register) & 0xFF,
        (int(value) >> 8) & 0xFF,
        int(value) & 0xFF,
    ])

    crc = _crc16_modbus(frame)

    # Modbus RTU CRC: low byte first
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


class BaseLightController:
    def start(self):
        pass

    def stop(self):
        pass

    def set_brightness(self, light_id, brightness):
        pass

    def get_state(self):
        return {}


class NullLightController(BaseLightController):
    def start(self):
        print("[LIGHT] disabled")

    def stop(self):
        pass


class MockLightController(BaseLightController):
    def __init__(self, light_cfg):
        self.light_cfg = light_cfg or {}
        self.lights = self.light_cfg.get("lights", [])
        self.state = {}

        for item in self.lights:
            light_id = item.get("id")
            if not light_id:
                continue

            self.state[light_id] = {
                "camera_id": item.get("camera_id"),
                "brightness": int(item.get("brightness", 0)),
            }

    def start(self):
        print(f"[LIGHT] mock started channels={len(self.state)}")

        for light_id, info in self.state.items():
            print(
                f"[LIGHT] {light_id} "
                f"camera={info.get('camera_id')} "
                f"brightness={info.get('brightness')}"
            )

    def stop(self):
        print("[LIGHT] mock stopped")

    def set_brightness(self, light_id, brightness):
        if light_id not in self.state:
            print(f"[LIGHT] unknown light_id: {light_id}")
            return False

        brightness = max(0, min(100, int(brightness)))
        self.state[light_id]["brightness"] = brightness

        print(f"[LIGHT] mock set {light_id} brightness={brightness}")
        return True

    def get_state(self):
        return dict(self.state)


class Jk10VLightController(BaseLightController):
    """
    JK-10V-P01-CH
    - RS485 Modbus RTU
    - Function 0x06
    - Register 0x000A
    - value = volt * 1000
    """

    def __init__(self, light_cfg):
        self.light_cfg = light_cfg or {}
        self.port = str(self.light_cfg.get("port", "/dev/ttyUSB0"))
        self.slave_id = int(self.light_cfg.get("slave_id", 1))
        self.baudrate = int(self.light_cfg.get("baudrate", 9600))
        self.timeout = float(self.light_cfg.get("timeout", 0.5))
        self.register = int(self.light_cfg.get("register", 0x000A))

        self.min_percent = int(self.light_cfg.get("min_percent", 0))
        self.max_percent = int(self.light_cfg.get("max_percent", 100))

        self.shutdown_to_zero = bool(self.light_cfg.get("shutdown_to_zero", False))

        self.lights = self.light_cfg.get("lights", [])
        self.state = {}

        for item in self.lights:
            light_id = item.get("id")
            if not light_id:
                continue

            brightness = int(item.get("brightness", self.light_cfg.get("default_brightness", 70)))
            brightness = self._clamp_percent(brightness)

            self.state[light_id] = {
                "camera_id": item.get("camera_id"),
                "brightness": brightness,
                "voltage": self._percent_to_voltage(brightness),
                "ok": None,
                "reason": "",
            }

    def _clamp_percent(self, value):
        value = int(round(float(value)))
        value = max(self.min_percent, min(self.max_percent, value))
        value = max(0, min(100, value))
        return value

    def _percent_to_voltage(self, percent):
        percent = self._clamp_percent(percent)
        return float(percent) / 100.0 * 10.0

    def _write_voltage(self, voltage):
        if serial is None:
            return False, "PYSERIAL_NOT_INSTALLED"

        voltage = max(0.0, min(10.0, float(voltage)))
        value = int(round(voltage * 1000.0))

        frame = _build_write_single_register(
            slave_id=self.slave_id,
            register=self.register,
            value=value,
        )

        try:
            with serial.Serial(
                port=self.port,
                baudrate=self.baudrate,
                bytesize=8,
                parity=serial.PARITY_NONE,
                stopbits=1,
                timeout=self.timeout,
            ) as ser:
                ser.reset_input_buffer()
                ser.reset_output_buffer()

                ser.write(frame)
                ser.flush()

                time.sleep(0.05)
                resp = ser.read(8)

        except Exception as e:
            return False, f"SERIAL_ERROR:{e}"

        if not resp:
            return False, "NO_RESPONSE"

        if resp != frame:
            return False, f"BAD_RESPONSE:{resp.hex(' ')}"

        return True, "OK"

    def start(self):
        print(
            f"[LIGHT] JK-10V started "
            f"port={self.port} slave={self.slave_id} baudrate={self.baudrate}"
        )

        for light_id, info in self.state.items():
            brightness = int(info.get("brightness", 0))
            voltage = self._percent_to_voltage(brightness)

            ok, reason = self._write_voltage(voltage)

            info["brightness"] = brightness
            info["voltage"] = voltage
            info["ok"] = bool(ok)
            info["reason"] = reason

            print(
                f"[LIGHT] {light_id} "
                f"brightness={brightness}% "
                f"voltage={voltage:.2f}V "
                f"result={reason}"
            )

    def stop(self):
        if self.shutdown_to_zero:
            for light_id in list(self.state.keys()):
                self.set_brightness(light_id, 0)
            print("[LIGHT] JK-10V stopped: output=0V")
        else:
            print("[LIGHT] JK-10V stopped")

    def set_brightness(self, light_id, brightness):
        if light_id not in self.state:
            print(f"[LIGHT] unknown light_id: {light_id}")
            return False

        brightness = self._clamp_percent(brightness)
        voltage = self._percent_to_voltage(brightness)

        ok, reason = self._write_voltage(voltage)

        self.state[light_id]["brightness"] = brightness
        self.state[light_id]["voltage"] = voltage
        self.state[light_id]["ok"] = bool(ok)
        self.state[light_id]["reason"] = reason

        print(
            f"[LIGHT] set {light_id} "
            f"brightness={brightness}% "
            f"voltage={voltage:.2f}V "
            f"result={reason}"
        )

        return bool(ok)

    def get_state(self):
        return dict(self.state)


def create_light_controller_from_hardware_config(hardware_cfg):
    hardware_cfg = hardware_cfg or {}

    light_sets = hardware_cfg.get("light_sets", {})
    active_light_set = hardware_cfg.get("active_light_set")

    if not active_light_set:
        return NullLightController()

    light_cfg = light_sets.get(active_light_set)

    if not light_cfg:
        print(f"[LIGHT] active light set not found: {active_light_set}")
        return NullLightController()

    backend = str(light_cfg.get("backend", "none")).strip().lower()

    if backend == "mock":
        return MockLightController(light_cfg)

    if backend in ("jk_10v", "jk_10v_modbus", "rs485_0_10v"):
        return Jk10VLightController(light_cfg)

    if backend == "none":
        return NullLightController()

    print(f"[LIGHT] unsupported backend: {backend}")
    return NullLightController()