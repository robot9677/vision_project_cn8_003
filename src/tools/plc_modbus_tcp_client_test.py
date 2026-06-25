import socket
import struct
import time


PLC_IP = "192.168.1.2"
PLC_PORT = 502

# PLC 화면 국번이 0이면 0부터 테스트
UNIT_ID = 0

# PLC D200을 Modbus Holding Register 200으로 가정
CMD_REG = 200
STATUS_REG = 201
RESULT_REG = 202


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
    header = struct.pack(">HHHB", tid, 0, len(pdu) + 1, unit_id)
    sock.sendall(header + pdu)

    rx_header = recv_exact(sock, 7)
    rx_tid, proto, length, rx_unit = struct.unpack(">HHHB", rx_header)

    rx_pdu = recv_exact(sock, length - 1)

    print(f"[RX] tid={rx_tid} unit={rx_unit} pdu={rx_pdu.hex(' ')}")

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

    print(f"[WRITE] reg={addr} value={value}")
    return modbus_request(sock, UNIT_ID, pdu)


def read_holding_registers(sock, start, count):
    # Function 03: Read Holding Registers
    pdu = bytes([
        3,
        (start >> 8) & 0xFF,
        start & 0xFF,
        (count >> 8) & 0xFF,
        count & 0xFF,
    ])

    print(f"[READ] start={start} count={count}")
    rx_pdu = modbus_request(sock, UNIT_ID, pdu)

    if len(rx_pdu) < 2 or rx_pdu[0] != 3:
        raise RuntimeError(f"invalid read response: {rx_pdu.hex(' ')}")

    byte_count = rx_pdu[1]
    data = rx_pdu[2:2 + byte_count]

    vals = []
    for i in range(0, len(data), 2):
        vals.append((data[i] << 8) | data[i + 1])

    return vals


def main():
    print("[MBTCP CLIENT TEST]")
    print(f"PLC IP   : {PLC_IP}")
    print(f"PLC Port : {PLC_PORT}")
    print(f"Unit ID  : {UNIT_ID}")
    print(f"WRITE D{CMD_REG}=1")
    print(f"READ  D{STATUS_REG}~D{RESULT_REG}")

    with socket.create_connection((PLC_IP, PLC_PORT), timeout=3.0) as sock:
        sock.settimeout(3.0)
        print("[CONNECT] OK")

        # PLC D200에 1 쓰기
        write_single_register(sock, CMD_REG, 1)

        time.sleep(0.2)

        # 확인용으로 D200~D202 읽기
        vals = read_holding_registers(sock, CMD_REG, 3)

        print("[RESULT]")
        print(f"D{CMD_REG} = {vals[0]}")
        print(f"D{STATUS_REG} = {vals[1]}")
        print(f"D{RESULT_REG} = {vals[2]}")


if __name__ == "__main__":
    main()