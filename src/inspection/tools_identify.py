from typing import Any, Dict, Tuple
import cv2
import numpy as np

from .toolchain import register_tool


def _ensure_gray8(crop: np.ndarray) -> np.ndarray:
    if crop is None:
        return None

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    if crop.dtype != np.uint8:
        crop = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    return crop


def _identify_qr(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    gray = _ensure_gray8(crop)
    if gray is None or gray.size == 0:
        return crop, {"qr_data": "", "has_qr": False}, False, "EMPTY_CROP"

    det = cv2.QRCodeDetector()
    data, points, _ = det.detectAndDecode(gray)

    data = (data or "").strip() if params.get("strip", True) else (data or "")
    has_qr = bool(data)

    expected = params.get("expected", None)
    expected_list = params.get("expected_list", None)

    ok = has_qr
    reason = "OK" if ok else "NO_QR"

    if ok and expected is not None:
        ok = (data == str(expected))
        reason = "OK" if ok else "QR_MISMATCH"

    if ok and expected_list is not None:
        try:
            allowed = [str(x) for x in expected_list]
            ok = data in allowed
            reason = "OK" if ok else "QR_NOT_ALLOWED"
        except Exception:
            ok = False
            reason = "CFG_ERROR"

    meta = {
        "qr_data": data,
        "has_qr": has_qr,
        "qr_text": data,
    }
    return crop, meta, bool(ok), reason


def register_identify_tools() -> None:
    register_tool("identify.qr", _identify_qr)
    register_tool("identify.barcode_qr", _identify_qr)