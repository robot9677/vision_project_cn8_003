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

def _clahe(crop, params, ctx):
    import cv2
    if crop is None or crop.size == 0:
        return crop, {}, False, "EMPTY"

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    clip = float(params.get("clip", 2.0))
    grid = int(params.get("grid", 8))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out = clahe.apply(gray)
    return out, {"clahe": 1}, True, "OK"

def register_enhance_tools() -> None:

    register_tool("enhance.noop", _noop)

    register_tool("enhance.threshold", _threshold)   

    register_tool("enhance.clahe", _clahe)