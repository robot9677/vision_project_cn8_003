import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .toolchain import register_tool

def _pattern_match(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:

    template_path = params.get("template")
    thresh = float(params.get("score_min", 0.8))

    if template_path is None:
        return crop, {"score":0.0}, False, "NO_TEMPLATE"

    tmpl = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl is None:
        return crop, {"score":0.0}, False, "LOAD_FAIL"

    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    res = cv2.matchTemplate(crop, tmpl, cv2.TM_CCOEFF_NORMED)
    _, score, _, _ = cv2.minMaxLoc(res)

    ok = score >= thresh

    return crop, {"score":float(score)}, ok, "OK" if ok else "LOW_SCORE"

def register_locate_tools():
    register_tool("locate.pattern_match", _pattern_match)