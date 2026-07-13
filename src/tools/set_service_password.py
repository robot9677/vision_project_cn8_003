#!/usr/bin/env python3
import getpass
import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CONFIG_PATH = PROJECT_ROOT / "data" / "config" / "plc_config.json"


def main():
    if not CONFIG_PATH.exists():
        raise RuntimeError(f"PLC config not found: {CONFIG_PATH}")

    pw1 = getpass.getpass("New SERVICE password: ")
    pw2 = getpass.getpass("Confirm password: ")
    if not pw1:
        raise RuntimeError("Password cannot be empty")
    if pw1 != pw2:
        raise RuntimeError("Passwords do not match")

    with CONFIG_PATH.open("r", encoding="utf-8-sig") as file:
        cfg = json.load(file)
    service_cfg = cfg.setdefault("service_panel", {})
    service_cfg["enabled"] = True
    service_cfg["password_sha256"] = hashlib.sha256(
        pw1.encode("utf-8")
    ).hexdigest()

    temp_path = CONFIG_PATH.with_suffix(".json.tmp")
    with temp_path.open("w", encoding="utf-8") as file:
        json.dump(cfg, file, ensure_ascii=False, indent=2)
    temp_path.replace(CONFIG_PATH)
    print(f"SERVICE password updated: {CONFIG_PATH}")
    print("Restart main_vp.py to apply the new password.")


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        print(f"[SET SERVICE PASSWORD] failed: {e}")
        raise SystemExit(1)
