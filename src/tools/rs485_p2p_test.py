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

rx_count = 0
tx_count = 0
ping_count = 0
last_rx_time = 0.0


def tx(ser, text):
    global tx_count

    data = text.encode("ascii") + LINE_ENDING
    ser.write(data)
    ser.flush()

    tx_count += 1
    print(f"[TX #{tx_count}] {data.hex(' ')}  {data!r}")


def handle_cmd(ser, text):
    global ping_count

    cmd = text.strip().upper()

    if cmd == "PING":
        ping_count += 1
        tx(ser, f"PONG,{ping_count}")

    elif cmd in ("START", "INSPECT", "1"):
        tx(ser, "BUSY")
        time.sleep(1.0)
        tx(ser, "OK")

    elif cmd in ("NG", "TEST_NG"):
        tx(ser, "NG")

    elif cmd in ("HELLO",):
        tx(ser, "PONG")

    else:
        tx(ser, "ERR")


def main():
    global rx_count, tx_count, ping_count, last_rx_time

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
    print("[RS485 P2P TEST] PLC heartbeat command: PING")
    print("[RS485 P2P TEST] response: PONG,count")
    print("[RS485 P2P TEST] Ctrl+C to stop")

    buf = bytearray()
    last_status_print = time.time()

    try:
        while True:
            data = ser.read(256)

            if data:
                rx_count += 1
                last_rx_time = time.time()

                print(f"[RX #{rx_count}] {data.hex(' ')}  {data!r}")

                # PLC에서 어떤 프레임이 오든 즉시 ACK 응답
                ack = b"ACK\r\n"
                ser.write(ack)
                ser.flush()

                tx_count += 1
                print(f"[TX #{tx_count}] {ack.hex(' ')}  {ack!r}")

            now = time.time()

            if now - last_status_print >= 2.0:
                if last_rx_time > 0:
                    age = now - last_rx_time
                    print(
                        f"[STATUS] rx_count={rx_count} tx_count={tx_count} "
                        f"ping_count={ping_count} last_rx_age={age:.1f}s"
                    )
                else:
                    print(
                        f"[STATUS] rx_count={rx_count} tx_count={tx_count} "
                        f"ping_count={ping_count} last_rx=NONE"
                    )

                last_status_print = now

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