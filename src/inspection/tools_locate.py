import cv2
import numpy as np
from typing import Dict, Any, Tuple
from .toolchain import register_tool

def _pattern_match(crop, params, ctx):
    template_path = params.get("template")
    score_min = float(params.get("score_min", 0.80))

    # 스케일 검색 범위
    s_min = float(params.get("scale_min", 0.85))
    s_max = float(params.get("scale_max", 1.15))
    s_step = float(params.get("scale_step", 0.05))

    if not template_path:
        return crop, {"score": 0.0}, False, "NO_TEMPLATE"

    tmpl0 = cv2.imread(template_path, cv2.IMREAD_GRAYSCALE)
    if tmpl0 is None:
        return crop, {"score": 0.0}, False, "LOAD_FAIL"

    if crop is None or crop.size == 0:
        return crop, {"score": 0.0}, False, "EMPTY_CROP"

    crop_g = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop

    best = -1.0
    best_s = 1.0

    s = s_min
    while s <= s_max + 1e-9:
        tw = int(tmpl0.shape[1] * s)
        th = int(tmpl0.shape[0] * s)
        if tw < 5 or th < 5:
            s += s_step
            continue

        tmpl = cv2.resize(tmpl0, (tw, th), interpolation=cv2.INTER_AREA)

        if crop_g.shape[0] < tmpl.shape[0] or crop_g.shape[1] < tmpl.shape[1]:
            s += s_step
            continue

        res = cv2.matchTemplate(crop_g, tmpl, cv2.TM_CCOEFF_NORMED)
        _, sc, _, _ = cv2.minMaxLoc(res)

        if sc > best:
            best = float(sc)
            best_s = float(s)

        s += s_step

    ok = best >= score_min
    return crop, {"score": best, "scale": best_s}, bool(ok), ("OK" if ok else "LOW_SCORE")

def register_locate_tools() -> None:
    register_tool("locate.pattern_match", _pattern_match)
    register_tool("locate.match", _pattern_match)