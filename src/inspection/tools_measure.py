# src/inspection/tools_measure.py
import cv2
import numpy as np
from typing import Any, Dict, Tuple
from .toolchain import register_tool

def _edge_energy(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    """
    params:
      - ksize: 3|5 (default 3)
      - sigma: float (default 0)  # 0이면 자동
      - thresh_min: float (optional)  # 합격 최소
      - thresh_max: float (optional)  # 합격 최대
    """
    if crop is None or crop.size == 0:
        return crop, {"edge_energy": 0.0}, False, "EMPTY_CROP"

    # ensure gray8
    if crop.dtype != np.uint8:
        gray = cv2.normalize(crop, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        gray = crop if crop.ndim == 2 else cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)

    ksize = int(params.get("ksize", 3))
    ksize = 3 if ksize not in (3, 5) else ksize
    sigma = float(params.get("sigma", 0))

    blur = cv2.GaussianBlur(gray, (ksize, ksize), sigmaX=sigma)

    gx = cv2.Sobel(blur, cv2.CV_32F, 1, 0, ksize=3)
    gy = cv2.Sobel(blur, cv2.CV_32F, 0, 1, ksize=3)

    mag = cv2.magnitude(gx, gy)
    energy = float(np.mean(mag))

    meta = {"edge_energy": energy}

    tmin = params.get("thresh_min", None)
    tmax = params.get("thresh_max", None)

    ok = True
    reason = "OK"
    if tmin is not None and energy < float(tmin):
        ok = False
        reason = "EDGE_LOW"
    if tmax is not None and energy > float(tmax):
        ok = False
        reason = "EDGE_HIGH"

    return crop, meta, bool(ok), reason

def register_measure_tools() -> None:
    register_tool("measure.edge_energy", _edge_energy)
    register_tool("measure.edge", _edge_energy)