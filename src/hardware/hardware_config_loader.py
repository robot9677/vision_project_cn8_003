import json
import os
from typing import Any, Dict


DEFAULT_HARDWARE_CONFIG: Dict[str, Any] = {
    "active_camera_set": "b0429_single",
    "active_light_set": "mock_single",
    "camera_sets": {
        "b0429_single": {
            "mode": "single",
            "backend": "gstreamer",
            "cameras": [
                {
                    "id": "cam1",
                    "name": "B0429_AR0234",
                    "device": "/dev/video0",
                    "width": 1280,
                    "height": 720,
                    "fps": 30,
                    "pipeline_type": "b0429_gray16_to_gray8",
                    "input_format": "GRAY16_LE",
                    "output_format": "GRAY8",
                    "camera_profile": "default",
                }
            ],
        }
    },
    "light_sets": {
        "mock_single": {
            "backend": "mock",
            "lights": [
                {
                    "id": "light1",
                    "camera_id": "cam1",
                    "brightness": 70,
                }
            ],
        }
    },
}


def load_hardware_config(path: str) -> Dict[str, Any]:
    if not os.path.exists(path):
        print(f"[HW CONFIG] not found, use default: {path}")
        return DEFAULT_HARDWARE_CONFIG.copy()

    with open(path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    if not isinstance(cfg, dict):
        raise ValueError(f"hardware_config must be object: {path}")

    return cfg