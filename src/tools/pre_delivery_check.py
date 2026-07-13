#!/usr/bin/env python3
"""Non-invasive pre-delivery configuration and source check.

The tool does not open the camera, serial port, or light controller. It can be
run while the vision application is stopped without changing runtime data.
"""

import compileall
import json
import os
import sys
from datetime import datetime
from typing import Any, Dict, List


PROJECT_ROOT = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..")
)
SRC_ROOT = os.path.join(PROJECT_ROOT, "src")
DATA_ROOT = os.path.join(PROJECT_ROOT, "data")
CONFIG_ROOT = os.path.join(DATA_ROOT, "config")
ROI_ROOT = os.path.join(DATA_ROOT, "roi")
LOG_ROOT = os.path.join(DATA_ROOT, "logs")


class CheckReport:
    def __init__(self):
        self.items: List[Dict[str, Any]] = []

    def add(self, level: str, name: str, detail: str):
        level = str(level).upper()
        self.items.append({"level": level, "name": name, "detail": detail})
        print(f"[{level:<4}] {name}: {detail}")

    def ok(self, name: str, detail: str):
        self.add("PASS", name, detail)

    def warn(self, name: str, detail: str):
        self.add("WARN", name, detail)

    def fail(self, name: str, detail: str):
        self.add("FAIL", name, detail)

    @property
    def fail_count(self) -> int:
        return sum(1 for item in self.items if item["level"] == "FAIL")

    @property
    def warn_count(self) -> int:
        return sum(1 for item in self.items if item["level"] == "WARN")


def _load_json(path: str):
    with open(path, "r", encoding="utf-8-sig") as file:
        return json.load(file)


def _check_json(report: CheckReport, label: str, path: str):
    try:
        value = _load_json(path)
        if not isinstance(value, dict):
            report.fail(label, f"root must be object: {path}")
            return None
        report.ok(label, path)
        return value
    except Exception as error:
        report.fail(label, f"{path}: {error}")
        return None


def _check_file(report: CheckReport, label: str, path: str, required=True):
    if os.path.isfile(path):
        report.ok(label, path)
        return True
    if required:
        report.fail(label, f"not found: {path}")
    else:
        report.warn(label, f"not found: {path}")
    return False


def main() -> int:
    report = CheckReport()
    print("=" * 78)
    print(" DAOL VISION PRE-DELIVERY CHECK (NON-INVASIVE)")
    print("=" * 78)
    print(f"Project: {PROJECT_ROOT}")

    compile_ok = compileall.compile_dir(SRC_ROOT, quiet=1)
    if compile_ok:
        report.ok("Python compile", "all src/*.py compiled")
    else:
        report.fail("Python compile", "one or more source files failed")

    runtime_path = os.path.join(ROI_ROOT, "runtime_config.json")
    hardware_path = os.path.join(CONFIG_ROOT, "hardware_config.json")
    plc_path = os.path.join(CONFIG_ROOT, "plc_config.json")

    runtime_cfg = _check_json(report, "Runtime config", runtime_path)
    hardware_cfg = _check_json(report, "Hardware config", hardware_path)
    plc_cfg = _check_json(report, "PLC config", plc_path)

    if runtime_cfg:
        profile_name = str(runtime_cfg.get("profile_name", "") or "").strip()
        if profile_name:
            profile_path = os.path.join(
                ROI_ROOT,
                "profiles",
                f"{profile_name}_profile.json",
            )
            profile_cfg = _check_json(report, "Product profile", profile_path)

            profile_roi_path = os.path.join(
                ROI_ROOT,
                "profiles",
                f"{profile_name}_roi.json",
            )
            if not _check_file(
                report,
                "Profile ROI runtime file",
                profile_roi_path,
                required=False,
            ):
                _check_file(
                    report,
                    "Fallback ROI file",
                    os.path.join(ROI_ROOT, "roi.json"),
                    required=True,
                )

            if profile_cfg:
                recipe_name = str(
                    profile_cfg.get("recipe_name", "tape_presence") or "tape_presence"
                ).strip()
                recipe_path = os.path.join(
                    ROI_ROOT,
                    "recipes",
                    f"{recipe_name}.json",
                )
                _check_json(report, "Active recipe", recipe_path)
        else:
            report.warn("Product profile", "profile_name is empty; fallback files used")

        snapshot_keep = int(runtime_cfg.get("snapshot_keep", 10))
        if snapshot_keep <= 20:
            report.ok("Dataset retention", f"keep={snapshot_keep} capture groups")
        else:
            report.warn("Dataset retention", f"keep={snapshot_keep} is relatively high")

    if hardware_cfg:
        camera_set_name = str(hardware_cfg.get("active_camera_set", "") or "")
        camera_set = (hardware_cfg.get("camera_sets", {}) or {}).get(camera_set_name, {})
        cameras = camera_set.get("cameras", []) if isinstance(camera_set, dict) else []
        if cameras and isinstance(cameras[0], dict):
            camera = cameras[0]
            device = str(camera.get("device", "") or "")
            pipeline = str(camera.get("pipeline_type", "") or "")
            report.ok("Active camera", f"{camera_set_name}, {pipeline}, {device}")
            if device and os.path.exists(device):
                report.ok("Camera device node", device)
            else:
                report.warn("Camera device node", f"not found now: {device}")
        else:
            report.fail("Active camera", f"invalid set: {camera_set_name}")

        light_set_name = str(hardware_cfg.get("active_light_set", "") or "")
        light_set = (hardware_cfg.get("light_sets", {}) or {}).get(light_set_name, {})
        if isinstance(light_set, dict) and light_set:
            idle = int((light_set.get("spot_inspect", {}) or {}).get("idle_brightness", -1))
            inspect = int((light_set.get("spot_inspect", {}) or {}).get("inspect_brightness", -1))
            report.ok("Active light", f"{light_set_name}, idle={idle}%, inspect={inspect}%")
            light_port = str(light_set.get("port", "") or "")
            if light_port and os.path.exists(light_port):
                report.ok("Light serial path", light_port)
            elif light_port:
                report.warn("Light serial path", f"not found now: {light_port}")
        else:
            report.fail("Active light", f"invalid set: {light_set_name}")

    if plc_cfg:
        serial_cfg = plc_cfg.get("serial", {}) or {}
        plc_port = str(serial_cfg.get("port", "") or "")
        slave_id = int((plc_cfg.get("modbus", {}) or {}).get("slave_id", 0))
        regs = plc_cfg.get("registers", {}) or {}
        report.ok(
            "PLC protocol",
            f"port={plc_port}, slave={slave_id}, regs={regs}",
        )
        if plc_port and os.path.exists(plc_port):
            report.ok("PLC serial device", plc_port)
        else:
            report.warn("PLC serial device", f"not found now: {plc_port}")

        service_cfg = plc_cfg.get("service_panel", {}) or {}
        password_hash = str(service_cfg.get("password_sha256", "") or "")
        if len(password_hash) == 64 and all(
            char in "0123456789abcdefABCDEF" for char in password_hash
        ):
            report.ok("Service password", "fixed SHA-256 configured")
        else:
            report.fail("Service password", "password_sha256 must be 64 hex characters")

        diagnostics = plc_cfg.get("diagnostics", {}) or {}
        if bool(diagnostics.get("enabled", False)):
            report.ok(
                "Diagnostics retention",
                (
                    f"errors={diagnostics.get('error_keep_files')} "
                    f"tests={diagnostics.get('test_keep_files')} "
                    f"normal={diagnostics.get('normal_keep_files')} "
                    f"days={diagnostics.get('keep_days')}"
                ),
            )
        else:
            report.warn("Diagnostics retention", "diagnostics disabled")

    autorun_script = os.path.join(SRC_ROOT, "tools", "start_vision_autorun.sh")
    _check_file(report, "Autostart RUN launcher", autorun_script, required=True)

    os.makedirs(os.path.join(LOG_ROOT, "diagnostics"), exist_ok=True)
    report_path = os.path.join(
        LOG_ROOT,
        "diagnostics",
        f"pre_delivery_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
    )
    payload = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "project_root": PROJECT_ROOT,
        "fail_count": report.fail_count,
        "warn_count": report.warn_count,
        "items": report.items,
    }
    with open(report_path, "w", encoding="utf-8") as file:
        json.dump(payload, file, ensure_ascii=False, indent=2)

    print("-" * 78)
    print(
        f"RESULT: FAIL={report.fail_count} WARN={report.warn_count} "
        f"REPORT={report_path}"
    )
    return 1 if report.fail_count else 0


if __name__ == "__main__":
    sys.exit(main())
