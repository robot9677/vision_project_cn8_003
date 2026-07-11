import json
import os
from datetime import datetime
from typing import Any, Dict, Optional


ERROR_NAMES = {
    0: "NO_ERROR",

    10: "CAMERA_INIT_FAILED",
    11: "CAMERA_FRAME_TIMEOUT",

    20: "LIGHT_START_FAILED",
    21: "LIGHT_COMMUNICATION_FAILED",

    30: "RECIPE_LOAD_FAILED",
    31: "ROI_CONFIG_ERROR",

    40: "INSPECTION_FAILED",
    41: "INSPECTION_EXCEPTION",

    50: "COMMAND_SEQUENCE_ERROR",
    51: "EMERGENCY_STOP",

    60: "VISION_RESET_FAILED",

    70: "PLC_SERIAL_OPEN_FAILED",
    71: "PLC_SERIAL_COMMUNICATION_LOST",

    90: "SHUTDOWN_FAILED",
}


def get_error_name(error_code: int) -> str:
    return ERROR_NAMES.get(int(error_code), "UNDEFINED_ERROR")


def _json_safe(value: Any) -> Any:
    if value is None:
        return None

    if isinstance(value, (str, int, float, bool)):
        return value

    if isinstance(value, dict):
        return {
            str(key): _json_safe(item)
            for key, item in value.items()
        }

    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]

    return str(value)


def save_plc_error_log(
    logs_root: str,
    event_type: str,
    error_code: int,
    message: str,
    plc_snapshot: Optional[Dict[str, Any]] = None,
    vision_snapshot: Optional[Dict[str, Any]] = None,
    exception: Optional[BaseException] = None,
) -> Optional[str]:

    now = datetime.now()

    log_dir = os.path.join(
        logs_root,
        "plc_errors",
        now.strftime("%Y%m%d"),
    )

    try:
        os.makedirs(log_dir, exist_ok=True)

        timestamp_text = now.strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
        filename_time = now.strftime("%H%M%S_%f")[:-3]

        error_code = int(error_code)

        log_data = {
            "schema_version": 1,
            "timestamp": timestamp_text,
            "event_type": str(event_type),
            "error": {
                "code": error_code,
                "name": get_error_name(error_code),
                "message": str(message or ""),
            },
            "plc_state": _json_safe(plc_snapshot or {}),
            "vision_state": _json_safe(vision_snapshot or {}),
            "exception": None,
        }

        if exception is not None:
            log_data["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "repr": repr(exception),
            }

        filename = (
            f"{filename_time}_"
            f"{error_code:02d}_"
            f"{get_error_name(error_code)}.json"
        )

        final_path = os.path.join(log_dir, filename)
        temp_path = final_path + ".tmp"

        with open(temp_path, "w", encoding="utf-8") as file:
            json.dump(
                log_data,
                file,
                ensure_ascii=False,
                indent=2,
            )

        os.replace(temp_path, final_path)

        print(f"[PLC ERROR LOG] saved: {final_path}")
        return final_path

    except Exception as log_error:
        print(f"[PLC ERROR LOG] save failed: {log_error}")
        return None