import socket
import struct
import threading
import time


HOST = "0.0.0.0"
PORT = 1502

UNIT_ID = 1

# PLC에서 D200/D201/D202로 접근한다고 가정
COMMAND_REG = 200
STATUS_REG = 201
RESULT_REG = 202

REG_COUNT = 300
regs = [0] * REG_COUNT
lock = threading.Lock()


def set_reg(addr, value):
    with lock:
        if 0 <= addr < len(regs):
            regs[addr] = int(value) & 0xFFFF


def get_reg(addr):
    with lock:
        if 0 <= addr < len(regs):
            return int(regs[addr])
        return 0


def recv_exact(conn, n):
    data = bytearray()
    while len(data) < n:
        chunk = conn.recv(n - len(data))
        if not chunk:
            return None
        data.extend(chunk)
    return bytes(data)


def make_exception(func, code):
    return bytes([func | 0x80, code])


def handle_read_registers(func, pdu):
    if len(pdu) < 5:
        return make_exception(func, 3)

    start = (pdu[1] << 8) | pdu[2]
    qty = (pdu[3] << 8) | pdu[4]

    print(f"[MBTCP] READ func={func} start={start} qty={qty}")

    if qty <= 0 or qty > 32:
        return make_exception(func, 3)

    if start < 0 or start + qty > len(regs):
        return make_exception(func, 2)

    data = bytearray()
    with lock:
        vals = regs[start:start + qty]

    for v in vals:
        data.append((v >> 8) & 0xFF)
        data.append(v & 0xFF)

    return bytes([func, len(data)]) + bytes(data)


def handle_write_single(func, pdu):
    if len(pdu) < 5:
        return make_exception(func, 3)

    addr = (pdu[1] << 8) | pdu[2]
    value = (pdu[3] << 8) | pdu[4]

    print(f"[MBTCP] WRITE_SINGLE addr={addr} value={value}")

    if addr < 0 or addr >= len(regs):
        return make_exception(func, 2)

    set_reg(addr, value)

    if addr == COMMAND_REG and value == 1:
        print("[MBTCP] COMMAND D200=1 received")
        set_reg(COMMAND_REG, 0)

        set_reg(STATUS_REG, 1)
        set_reg(RESULT_REG, 0)
        print("[MBTCP] STATUS D201=1 BUSY")

        time.sleep(1.0)

        set_reg(STATUS_REG, 2)
        set_reg(RESULT_REG, 1)
        print("[MBTCP] DONE D201=2, RESULT D202=1 OK")

    return pdu[:5]


def handle_write_multiple(func, pdu):
    if len(pdu) < 6:
        return make_exception(func, 3)

    start = (pdu[1] << 8) | pdu[2]
    qty = (pdu[3] << 8) | pdu[4]
    byte_count = pdu[5]

    print(f"[MBTCP] WRITE_MULTIPLE start={start} qty={qty}")

    if qty <= 0 or qty > 32 or byte_count != qty * 2:
        return make_exception(func, 3)

    if len(pdu) < 6 + byte_count:
        return make_exception(func, 3)

    if start < 0 or start + qty > len(regs):
        return make_exception(func, 2)

    pos = 6
    for i in range(qty):
        value = (pdu[pos] << 8) | pdu[pos + 1]
        set_reg(start + i, value)
        print(f"[MBTCP]   reg[{start + i}]={value}")
        pos += 2

    if start <= COMMAND_REG < start + qty:
        if get_reg(COMMAND_REG) == 1:
            print("[MBTCP] COMMAND D200=1 received")
            set_reg(COMMAND_REG, 0)

            set_reg(STATUS_REG, 1)
            set_reg(RESULT_REG, 0)
            print("[MBTCP] STATUS D201=1 BUSY")

            time.sleep(1.0)

            set_reg(STATUS_REG, 2)
            set_reg(RESULT_REG, 1)
            print("[MBTCP] DONE D201=2, RESULT D202=1 OK")

    return bytes([
        func,
        (start >> 8) & 0xFF,
        start & 0xFF,
        (qty >> 8) & 0xFF,
        qty & 0xFF,
    ])


def handle_client(conn, addr):
    print(f"[MBTCP] client connected: {addr}")

    try:
        while True:
            header = recv_exact(conn, 7)
            if not header:
                break

            transaction_id, protocol_id, length, unit_id = struct.unpack(">HHHB", header)

            if protocol_id != 0:
                print(f"[MBTCP] invalid protocol_id={protocol_id}")
                continue

            pdu = recv_exact(conn, length - 1)
            if not pdu:
                break

            func = pdu[0]

            print(
                f"[MBTCP] RX tid={transaction_id} unit={unit_id} "
                f"func={func} pdu={pdu.hex(' ')}"
            )

            if unit_id not in (UNIT_ID, 0):
                print(f"[MBTCP] ignore unit_id={unit_id}")
                continue

            if func in (3, 4):
                response_pdu = handle_read_registers(func, pdu)
            elif func == 6:
                response_pdu = handle_write_single(func, pdu)
            elif func == 16:
                response_pdu = handle_write_multiple(func, pdu)
            else:
                response_pdu = make_exception(func, 1)

            response_header = struct.pack(
                ">HHHB",
                transaction_id,
                0,
                len(response_pdu) + 1,
                unit_id
            )

            conn.sendall(response_header + response_pdu)

    except Exception as e:
        print(f"[MBTCP] client error {addr}: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass
        print(f"[MBTCP] client disconnected: {addr}")


def main():
    set_reg(COMMAND_REG, 0)
    set_reg(STATUS_REG, 0)
    set_reg(RESULT_REG, 0)

    print("[MBTCP] Modbus TCP test server start")
    print(f"[MBTCP] listen {HOST}:{PORT}")
    print("[MBTCP] Jetson IP: 192.168.1.3")
    print(f"[MBTCP] Unit ID: {UNIT_ID}")
    print(f"[MBTCP] D{COMMAND_REG}=command, D{STATUS_REG}=status, D{RESULT_REG}=result")

    server = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    server.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
    server.bind((HOST, PORT))
    server.listen(5)

    while True:
        conn, addr = server.accept()
        t = threading.Thread(target=handle_client, args=(conn, addr), daemon=True)
        t.start()


if __name__ == "__main__":
    main()