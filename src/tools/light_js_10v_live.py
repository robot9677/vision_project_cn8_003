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
    return frame + bytes([crc & 0xFF, (crc >> 8) & 0xFF])


def clamp(v, lo, hi):
    return max(lo, min(hi, v))


def send_voltage(port, slave_id, baudrate, voltage, timeout=0.3):
    voltage = clamp(float(voltage), 0.0, 10.0)
    value = int(round(voltage * 1000.0))

    frame = build_write_single_register(
        slave_id=slave_id,
        register=0x000A,
        value=value,
    )

    try:
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

            time.sleep(0.08)
            resp = ser.read(8)

    except Exception as e:
        return False, f"SERIAL_ERROR:{e}"

    if not resp:
        # 실제 전압 출력은 적용되지만 응답을 못 읽는 경우가 있으므로 OK 취급
        return True, "OK_NO_RESPONSE"

    if resp == frame:
        return True, "OK"

    return False, f"BAD_RESPONSE:{resp.hex(' ')}"


def apply_percent(port, slave_id, baudrate, percent):
    percent = int(round(clamp(float(percent), 0.0, 100.0)))
    voltage = percent / 100.0 * 10.0

    ok, reason = send_voltage(
        port=port,
        slave_id=slave_id,
        baudrate=baudrate,
        voltage=voltage,
    )

    print(
        f"[LIGHT LIVE] brightness={percent:3d}% "
        f"voltage={voltage:4.2f}V "
        f"result={reason}"
    )

    return percent, ok, reason


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--port", default="/dev/ttyUSB0")
    ap.add_argument("--slave", type=int, default=1)
    ap.add_argument("--baud", type=int, default=9600)
    ap.add_argument("--start", type=int, default=70)
    ap.add_argument("--step", type=int, default=5)
    args = ap.parse_args()

    percent = int(clamp(args.start, 0, 100))
    step = int(clamp(args.step, 1, 50))

    print("")
    print("[LIGHT LIVE] JK-10V live brightness tuner")
    print(f"[LIGHT LIVE] port={args.port} slave={args.slave} baud={args.baud}")
    print("")
    print("commands:")
    print("  +       : brightness up")
    print("  -       : brightness down")
    print("  0~100   : set brightness percent")
    print("  v0~v10  : set voltage directly, example v6.5")
    print("  r       : resend current value")
    print("  q       : quit")
    print("")

    percent, _, _ = apply_percent(args.port, args.slave, args.baud, percent)

    while True:
        try:
            cmd = input(f"light[{percent}%]> ").strip().lower()
        except KeyboardInterrupt:
            print("")
            break

        if cmd in ("q", "quit", "exit"):
            break

        if cmd == "":
            cmd = "r"

        if cmd in ("+", "up"):
            percent += step

        elif cmd in ("-", "down"):
            percent -= step

        elif cmd.startswith("+") and len(cmd) > 1:
            try:
                percent += int(float(cmd[1:]))
            except Exception:
                print("[LIGHT LIVE] invalid command")
                continue

        elif cmd.startswith("-") and len(cmd) > 1:
            try:
                percent -= int(float(cmd[1:]))
            except Exception:
                print("[LIGHT LIVE] invalid command")
                continue

        elif cmd.startswith("v"):
            try:
                voltage = float(cmd[1:])
                voltage = clamp(voltage, 0.0, 10.0)
                percent = int(round(voltage * 10.0))
            except Exception:
                print("[LIGHT LIVE] invalid voltage")
                continue

        elif cmd == "r":
            pass

        else:
            try:
                percent = int(round(float(cmd.replace("%", ""))))
            except Exception:
                print("[LIGHT LIVE] invalid command")
                continue

        percent = int(clamp(percent, 0, 100))
        percent, _, _ = apply_percent(args.port, args.slave, args.baud, percent)

    print(f"[LIGHT LIVE] final brightness={percent}%")
    print("[LIGHT LIVE] quit")


if __name__ == "__main__":
    main()