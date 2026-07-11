import os
import select
import sys
import time

SRC_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if SRC_DIR not in sys.path:
    sys.path.insert(0, SRC_DIR)

from app.app_paths import PLC_CONFIG_PATH
from plc.plc_config_loader import load_plc_config
from plc.plc_controller import create_plc_controller


def main():
    cfg = load_plc_config(PLC_CONFIG_PATH)

    if not bool(cfg.get("enabled", False)):
        print("[PLC TEST] plc_config.json enabled=false")
        print("[PLC TEST] set enabled=true first")
        return

    plc = create_plc_controller(cfg)
    plc.start()

    print("")
    print("===== PLC MANUAL TEST =====")
    print("i : set READY  (D201=0)")
    print("b : set BUSY   (D201=1)")
    print("o : set OK     (D201=2, D202=1)")
    print("n : set NG     (D201=2, D202=2)")
    print("e : set ERROR  (D201=3)")
    print("r : reset      (D201=0, D202=0)")
    print("s : snapshot")
    print("q : quit")
    print("===========================")
    print("D200 commands: 1=prepare, 2=inspect, 3=reset, 8=shutdown, 9=emergency")
    print("")

    try:
        while True:
            cmd = plc.poll_command()
            if cmd is not None:
                print(f"[PLC TEST] command received: {cmd}")
                if cmd == "inspect":
                    print("[PLC TEST] manually press o/n to return result")

            if select.select([sys.stdin], [], [], 0.1)[0]:
                key = sys.stdin.readline().strip().lower()

                if key == "q":
                    break

                elif key == "i":
                    plc.set_idle()
                    print("[PLC TEST] set IDLE")

                elif key == "b":
                    plc.set_busy()
                    print("[PLC TEST] set BUSY")

                elif key == "o":
                    plc.set_done(True)
                    print("[PLC TEST] set OK")

                elif key == "n":
                    plc.set_done(False)
                    print("[PLC TEST] set NG")

                elif key == "e":
                    plc.set_error()
                    print("[PLC TEST] set ERROR")

                elif key == "r":
                    plc.reset_to_ready(reset_heartbeat=False)
                    print("[PLC TEST] reset READY")

                elif key == "s":
                    print("[PLC TEST]", plc.get_state_snapshot())

            time.sleep(0.02)

    finally:
        plc.stop()


if __name__ == "__main__":
    main()