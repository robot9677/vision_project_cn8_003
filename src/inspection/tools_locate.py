import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .toolchain import register_tool

def _pattern_match(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    template_path = params.get("template")
    score_min = float(params.get("score_min", 0.85))

    if not template_path:
        return crop, {"score": 0.0}, False, "NO_TEMPLATE"

    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        return crop, {"score": 0.0}, False, "LOAD_FAIL"

    if crop is None or crop.size == 0:
        return crop, {"score": 0.0}, False, "EMPTY_CROP"

    if crop.ndim == 3:
        crop_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        crop_g = crop

    # crop이 template보다 작으면 실패
    if crop_g.shape[0] < tmpl.shape[0] or crop_g.shape[1] < tmpl.shape[1]:
        return crop, {"score": 0.0}, False, "CROP_TOO_SMALL"

    res = cv2.matchTemplate(crop_g, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)

    ok = (score >= score_min)
    return crop, {"score": float(score)}, bool(ok), ("OK" if ok else "LOW_SCORE")

def register_locate_tools() -> None:
    register_tool("locate.pattern_match", _pattern_match)
    register_tool("locate.match", _pattern_match)