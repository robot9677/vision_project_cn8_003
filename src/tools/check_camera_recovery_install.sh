#!/bin/bash
set -euo pipefail

PROJECT_ROOT="${1:-/home/robot96/vision_project}"
cd "${PROJECT_ROOT}/src"

python3 -m py_compile \
  main_vp.py \
  capture/camera_factory.py \
  capture/camera_process.py \
  ui/service_panel.py \
  tools/camera_recovery_stress.py

echo "[OK] Python syntax"

python3 - <<'PY'
import json
from hardware.hardware_config_loader import load_hardware_config
from capture.camera_factory import create_camera_from_hardware_config
from app.app_paths import HARDWARE_CONFIG_PATH
cfg = load_hardware_config(HARDWARE_CONFIG_PATH)
cam, info = create_camera_from_hardware_config(cfg)
print('[CHECK] pipeline_type=', info.get('pipeline_type'))
print('[CHECK] process_isolation_enabled=', info.get('process_isolation_enabled'))
print('[CHECK] camera_class=', type(cam).__name__)
cam.release()
if info.get('pipeline_type') == 'nvargus_bgr' and not info.get('process_isolation_enabled'):
    raise SystemExit('IMX477 Argus process isolation is not enabled')
PY

sudo -n /usr/bin/systemctl is-active nvargus-daemon >/dev/null || RC=$?
RC="${RC:-0}"
if [ "${RC}" -ne 0 ] && [ "${RC}" -ne 3 ]; then
  echo "[FAIL] nvargus-daemon sudoers permission rc=${RC}"
  exit 1
fi

echo "[OK] nvargus-daemon recovery permission"
echo "카메라 자동복구 설치 점검 완료"
