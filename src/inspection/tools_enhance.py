# src/inspection/tools_enhance.py
import cv2
from typing import Any, Dict, Tuple
import numpy as np
from .toolchain import register_tool

def _noop(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    return crop, {"enhance": "noop"}, True, "OK"

def _threshold(crop, params, ctx):
    mode = params.get("mode", "otsu")

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    if mode == "otsu":
        _, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        t = params.get("th", 128)
        _, th = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

    white_ratio = float((th > 0).sum()) / th.size

    meta = {
        "white_ratio": white_ratio
    }

    return th, meta, True, "OK"

def register_enhance_tools() -> None:

    register_tool("enhance.noop", _noop)
    
    register_tool("enhance.threshold", _threshold)   # ← 여기
