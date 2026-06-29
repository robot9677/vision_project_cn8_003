#!/usr/bin/env python3
import argparse
import time
import serial


def modbus_crc16(data: bytes) -> int:
    crc = 0xFFFF
    for b in data:
        crc ^= b
        for _ in range(8):
            if crc & 0x0001:
                crc = (crc >> 1) ^ 0xA001
            else:
                crc >>= 1
    return crc & 0xFFFF


def build_write_single_register(slave_id: int, register: int, value: int) -> bytes:
    frame = bytes([
        slave_id & 0xFF,
        0x06,
        (register >> 8) & 0xFF,
        register & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF,
    ])

    crc = modbus_crc16(frame)

    # Modbus RTU CRC는 Low byte 먼저 전송
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def set_voltage(port: str, slave_id: int, baudrate: int, voltage: float, timeout: float = 0.5):
    voltage = max(0.0, min(10.0, float(voltage)))
    value = int(round(voltage * 1000.0))  # 0~10V => 0~10000

    frame = build_write_single_register(
        slave_id=slave_id,
        register=0x000A,
        value=value,
    )

    print(f"[JK-10V] port={port} slave={slave_id} baud={baudrate}")
    print(f"[JK-10V] voltage={voltage:.3f}V value={value}")
    print("[JK-10V] TX:", frame.hex(" "))

    with serial.Serial(
        port=port,
        baudrate=baudrate,
        bytesize=8,
        parity=serial.PARITY_NONE,
        stopbits=1,
        timeout=timeout,
    ) as ser:
        ser.reset_input_buffer()
        ser.reset_output_buffer()

        ser.write(frame)
        ser.flush()

        time.sleep(0.1)
        resp = ser.read(8)

    print("[JK-10V] RX:", resp.hex(" ") if resp else "<no response>")

    if not resp:
        return False, "NO_RESPONSE"

    if resp == frame:
        return True, "OK"

    return False, "BAD_RESPONSE"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--port",
        default="/dev/serial/by-id/usb-1a86_USB_Serial-if00-port0",
        help="RS485 serial port",
    )
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--volt", type=float, default=None)
    ap.add_argument("--percent", type=float, default=None)
    args = ap.parse_args()

    if args.volt is None and args.percent is None:
        raise SystemExit("use --volt 0~10 or --percent 0~100")

    if args.volt is not None:
        voltage = float(args.volt)
    else:
        pct = max(0.0, min(100.0, float(args.percent)))
        voltage = pct / 100.0 * 10.0

    ok, reason = set_voltage(
        port=args.port,
        slave_id=args.slave,
        baudrate=args.baud,
        voltage=voltage,
    )

    print(f"[JK-10V] result={reason}")
    raise SystemExit(0 if ok else 1)


if __name__ == "__main__":
    main()