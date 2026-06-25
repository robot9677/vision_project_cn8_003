import socket
import threading
import time


HOST = "0.0.0.0"
PORT = 5000


def handle_client(conn, addr):
    print(f"[LAN TEST] client connected: {addr}")

    try:
        conn.settimeout(1.0)

        while True:
            try:
                data = conn.recv(1024)
            except socket.timeout:
                continue

            if not data:
                break

            text = data.decode("utf-8", errors="replace").strip()
            cmd = text.upper()

            print(f"[LAN TEST] RX raw={data!r} text='{text}'")

            if cmd in ("PING", "HELLO"):
                conn.sendall(b"PONG\r\n")
                print("[LAN TEST] TX: PONG")

            elif cmd in ("START", "INSPECT", "1"):
                conn.sendall(b"BUSY\r\n")
                print("[LAN TEST] TX: BUSY")

                time.sleep(1.0)

                conn.sendall(b"OK\r\n")
                print("[LAN TEST] TX: OK")

            elif cmd in ("NG", "TEST_NG"):
                conn.sendall(b"NG\r\n")
                print("[LAN TEST] TX: NG")

            else:
                conn.sendall(b"ERR\r\n")
                print("[LAN TEST] TX: ERR")

    except Exception as e:
        print(f"[LAN TEST] client error {addr}: {e}")

    finally:
        try:
            conn.close()
        except Exception:
            pass

        print(f"[LAN TEST] client disconnected: {addr}")


def main():
    print("[LAN TEST] TCP server start")
    print(f"[LAN TEST] listen: {HOST}:{PORT}")
    print("[LAN TEST] Jetson IP: 192.168.1.3")
    print("[LAN TEST] PLC send: PING or START")

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