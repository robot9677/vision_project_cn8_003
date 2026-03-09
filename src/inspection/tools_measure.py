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

def _blob_count(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    """
    input img: binary(0/255) 권장 (enhance.threshold 출력)
    params:
      - polarity: "white"|"black" (default "white")  # 카운트할 블롭 색
      - expected: int (optional)  # 기대 개수
      - min_count/max_count: (optional)
      - area_min/area_max: (optional)
      - open: int (optional)  # morphological open kernel size (0이면 안함)
    """
    if img is None or img.size == 0:
        return img, {"blob_count": 0}, False, "EMPTY"

    # ensure uint8 binary
    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    _, bw = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY)

    polarity = str(params.get("polarity", "white")).lower()
    if polarity == "black":
        bw = cv2.bitwise_not(bw)

    k = int(params.get("open", 0))
    if k >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k, k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # stats[0]는 배경
    area_min = float(params.get("area_min", 0))
    area_max = float(params.get("area_max", 1e18))

    cnt = 0
    areas_all = []
    areas_kept = []

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        areas_all.append(area)
        if area_min <= area <= area_max:
            cnt += 1
            areas_kept.append(area)

    meta = {
        "blob_count": int(cnt),
        "blob_areas_all": areas_all,
        "blob_areas_kept": areas_kept,
    }

    expected = params.get("expected", None)
    min_count = params.get("min_count", None)
    max_count = params.get("max_count", None)

    ok = True
    reason = "OK"
    if expected is not None and cnt != int(expected):
        ok = False; reason = "COUNT_MISMATCH"
    if min_count is not None and cnt < int(min_count):
        ok = False; reason = "COUNT_LOW"
    if max_count is not None and cnt > int(max_count):
        ok = False; reason = "COUNT_HIGH"

    return bw, meta, bool(ok), reason

def register_measure_tools() -> None:
    register_tool("measure.edge_energy", _edge_energy)
    register_tool("measure.edge", _edge_energy)
    register_tool("measure.blob_count", _blob_count)