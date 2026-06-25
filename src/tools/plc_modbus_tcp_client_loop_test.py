import socket
import struct
import time


PLC_IP = "192.168.1.2"
PLC_PORT = 502

# PLC 화면 국번이 0이면 0
# 국번 1로 설정했으면 1로 변경
UNIT_ID = 0

# PLC 쪽 D 레지스터 주소
CMD_REG = 200
STATUS_REG = 201
RESULT_REG = 202

# 반복 주기
CYCLE_SEC = 1.0

_tid = 1


def next_tid():
    global _tid
    v = _tid
    _tid = (_tid + 1) & 0xFFFF
    if _tid == 0:
        _tid = 1
    return v


def recv_exact(sock, n):
    data = bytearray()
    while len(data) < n:
        chunk = sock.recv(n - len(data))
        if not chunk:
            raise RuntimeError("connection closed")
        data.extend(chunk)
    return bytes(data)


def modbus_request(sock, unit_id, pdu):
    tid = next_tid()

    # MBAP Header: Transaction ID, Protocol ID, Length, Unit ID
    header = struct.pack(">HHHB", tid, 0, len(pdu) + 1, unit_id)

    sock.sendall(header + pdu)

    rx_header = recv_exact(sock, 7)
    rx_tid, proto, length, rx_unit = struct.unpack(">HHHB", rx_header)

    rx_pdu = recv_exact(sock, length - 1)

    if rx_tid != tid:
        raise RuntimeError(f"TID mismatch tx={tid} rx={rx_tid}")

    if rx_pdu and (rx_pdu[0] & 0x80):
        err_code = rx_pdu[1] if len(rx_pdu) > 1 else -1
        raise RuntimeError(f"Modbus exception func={rx_pdu[0]:02X} code={err_code}")

    return rx_pdu


def write_single_register(sock, addr, value):
    # Function 06: Write Single Holding Register
    pdu = bytes([
        6,
        (addr >> 8) & 0xFF,
        addr & 0xFF,
        (value >> 8) & 0xFF,
        value & 0xFF,
    ])

    rx = modbus_request(sock, UNIT_ID, pdu)
    print(f"[WRITE] D{addr}={value} RX={rx.hex(' ')}")


def read_holding_registers(sock, start, count):
    # Function 03: Read Holding Registers
    pdu = bytes([
        3,
        (start >> 8) & 0xFF,
        start & 0xFF,
        (count >> 8) & 0xFF,
        count & 0xFF,
    ])

    rx = modbus_request(sock, UNIT_ID, pdu)

    if len(rx) < 2 or rx[0] != 3:
        raise RuntimeError(f"invalid read response: {rx.hex(' ')}")

    byte_count = rx[1]
    data = rx[2:2 + byte_count]

    vals = []
    for i in range(0, len(data), 2):
        vals.append((data[i] << 8) | data[i + 1])

    return vals


def run_loop():
    print("[MBTCP CLIENT LOOP TEST]")
    print(f"PLC IP   : {PLC_IP}")
    print(f"PLC Port : {PLC_PORT}")
    print(f"Unit ID  : {UNIT_ID}")
    print(f"WRITE    : D{CMD_REG} 1 -> 0 반복")
    print(f"READ     : D{CMD_REG}~D{RESULT_REG}")
    print("Ctrl+C to stop")

    while True:
        try:
            print("[CONNECT] trying...")

            with socket.create_connection((PLC_IP, PLC_PORT), timeout=3.0) as sock:
                sock.settimeout(3.0)
                print("[CONNECT] OK")

                cycle = 0

                while True:
                    cycle += 1
                    print(f"\n[CYCLE {cycle}]")

                    # 검사 시작 신호처럼 1 펄스 전송
                    write_single_register(sock, CMD_REG, 1)
                    time.sleep(0.2)

                    # 0으로 리셋
                    write_single_register(sock, CMD_REG, 0)
                    time.sleep(0.2)

                    # 확인용 읽기
                    vals = read_holding_registers(sock, CMD_REG, 3)

                    print(
                        f"[READ] "
                        f"D{CMD_REG}={vals[0]} "
                        f"D{STATUS_REG}={vals[1]} "
                        f"D{RESULT_REG}={vals[2]}"
                    )

                    time.sleep(CYCLE_SEC)

        except KeyboardInterrupt:
            print("\n[STOP]")
            break

        except Exception as e:
            print(f"[ERROR] {e}")
            print("[RETRY] reconnect after 2 sec")
            time.sleep(2.0)


if __name__ == "__main__":
    run_loop()