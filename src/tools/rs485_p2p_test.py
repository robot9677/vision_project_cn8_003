import time

try:
    import serial
except Exception:
    serial = None


PORT = "/dev/ttyUSB0"
BAUDRATE = 9600

BYTESIZE = 8
PARITY = "N"
STOPBITS = 1
TIMEOUT = 0.1

LINE_ENDING = b"\r\n"


def main():
    if serial is None:
        raise RuntimeError("pyserial is not installed")

    ser = serial.Serial(
        port=PORT,
        baudrate=BAUDRATE,
        bytesize=BYTESIZE,
        parity=PARITY,
        stopbits=STOPBITS,
        timeout=TIMEOUT,
    )

    print("[RS485 P2P TEST] started")
    print(f"[RS485 P2P TEST] port={PORT} baud={BAUDRATE} 8N1")
    print("[RS485 P2P TEST] waiting PLC command: START / INSPECT / 1")
    print("[RS485 P2P TEST] Ctrl+C to stop")

    buf = bytearray()

    try:
        while True:
            data = ser.read(256)

            if data:
                print(f"[RX RAW] {data.hex(' ')}  {data!r}")
                buf.extend(data)

                # 줄바꿈 기준 명령 처리
                while b"\n" in buf or b"\r" in buf:
                    cut_positions = []
                    for sep in (b"\n", b"\r"):
                        p = buf.find(sep)
                        if p >= 0:
                            cut_positions.append(p)

                    if not cut_positions:
                        break

                    pos = min(cut_positions)
                    line = bytes(buf[:pos]).strip()
                    del buf[:pos + 1]

                    if not line:
                        continue

                    text = line.decode("utf-8", errors="replace").strip()
                    cmd = text.upper()

                    print(f"[RX CMD] '{text}'")

                    if cmd in ("START", "INSPECT", "1"):
                        ser.write(b"BUSY" + LINE_ENDING)
                        ser.flush()
                        print("[TX] BUSY")

                        time.sleep(1.0)

                        ser.write(b"OK" + LINE_ENDING)
                        ser.flush()
                        print("[TX] OK")

                    elif cmd in ("PING", "HELLO"):
                        ser.write(b"PONG" + LINE_ENDING)
                        ser.flush()
                        print("[TX] PONG")

                    elif cmd in ("NG", "TEST_NG"):
                        ser.write(b"NG" + LINE_ENDING)
                        ser.flush()
                        print("[TX] NG")

                    else:
                        ser.write(b"ERR" + LINE_ENDING)
                        ser.flush()
                        print("[TX] ERR")

            time.sleep(0.01)

    except KeyboardInterrupt:
        print("\n[RS485 P2P TEST] stopped")

    finally:
        try:
            ser.close()
        except Exception:
            pass


if __name__ == "__main__":
    main()