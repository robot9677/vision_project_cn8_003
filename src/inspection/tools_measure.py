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
    
    zx, zy = 0, 0
    zone = params.get("count_zone")
    if isinstance(zone, (list, tuple)) and len(zone) == 4:
        x0, y0, zw, zh = [int(v) for v in zone]
        H, W = img8.shape[:2]
        x1 = max(0, min(x0, W - 1))
        y1 = max(0, min(y0, H - 1))
        x2 = max(x1 + 1, min(W, x1 + zw))
        y2 = max(y1 + 1, min(H, y1 + zh))
        img8 = img8[y1:y2, x1:x2]
        zx, zy = x1, y1

    _, bw = cv2.threshold(img8, 0, 255, cv2.THRESH_BINARY)

    polarity = str(params.get("polarity", "white")).lower()
    if polarity == "black":
        bw = cv2.bitwise_not(bw)

    k_close = int(params.get("close", 0))
    if k_close >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_close, k_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)

    k_open = int(params.get("open", 0))
    if k_open >= 3:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (k_open, k_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)
    # stats[0]는 배경
    area_min = float(params.get("area_min", 0))
    area_max = float(params.get("area_max", 1e18))

    cnt = 0
    areas_all = []
    areas_kept = []
    boxes_kept = []

    bh, bw_img = bw.shape[:2]

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])

        sx = int(stats[i, cv2.CC_STAT_LEFT])
        sy = int(stats[i, cv2.CC_STAT_TOP])
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])

        x = sx + zx
        y = sy + zy

        areas_all.append(area)

        # zone 내부 경계에 닿는 blob 제외
        if sx <= 0 or sy <= 0 or (sx + w) >= bw_img or (sy + h) >= bh:
            continue

        if area_min <= area <= area_max:
            cnt += 1
            areas_kept.append(area)
            boxes_kept.append([x, y, w, h])

    meta = {
        "blob_count": int(cnt),
        "blob_areas_all": areas_all,
        "blob_areas_kept": areas_kept,
        "blob_boxes_kept": boxes_kept,
        "num_labels": int(num - 1),
        "count_zone": params.get("count_zone"),
        "area_min": area_min,
        "area_max": area_max,
        "polarity": polarity,
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

def _dark_ratio(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    """
    params:
      - thresh: int (default 60)         # 이 값 이하를 dark 로 판단
      - min_ratio: float (optional)      # 합격 최소 dark 비율
      - max_ratio: float (optional)      # 합격 최대 dark 비율
      - invert: bool (default False)     # debug용 반전 표시
      - blur: int (default 0)            # 3,5... 가우시안 블러
    """
    if img is None or img.size == 0:
        return img, {"dark_ratio": 0.0}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    k = int(params.get("blur", 0))
    if k >= 3:
        if k % 2 == 0:
            k += 1
        img8 = cv2.GaussianBlur(img8, (k, k), 0)

    base_mode = str(params.get("thresh_mode", "fixed")).lower()

    if base_mode == "mean_offset":
        offset = float(params.get("offset", -8))
        thresh = int(np.clip(float(np.mean(img8)) + offset, 0, 255))
    else:
        thresh = int(params.get("thresh", 60))

    _, bw = cv2.threshold(img8, thresh, 255, cv2.THRESH_BINARY_INV)

    dark_ratio = float(np.count_nonzero(bw)) / float(bw.size) if bw.size else 0.0

    if bool(params.get("invert", False)):
        dbg = cv2.bitwise_not(bw)
    else:
        dbg = bw

    min_ratio = params.get("min_ratio", None)
    max_ratio = params.get("max_ratio", None)

    ok = True
    reason = "OK"

    if min_ratio is not None and dark_ratio < float(min_ratio):
        ok = False
        reason = "DARK_RATIO_LOW"

    if max_ratio is not None and dark_ratio > float(max_ratio):
        ok = False
        reason = "DARK_RATIO_HIGH"

    meta = {
        "dark_ratio": float(dark_ratio),
        "dark_thresh": int(thresh),
        "dark_mean": float(np.mean(img8)),
    }

    return dbg, meta, bool(ok), reason

def _bright_ratio(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    """
    params:
      - thresh: int (default 180)        # 이 값 이상을 bright 로 판단
      - min_ratio: float (optional)      # 합격 최소 bright 비율
      - max_ratio: float (optional)      # 합격 최대 bright 비율
      - invert: bool (default False)     # debug용 반전 표시
      - blur: int (default 0)            # 3,5... 가우시안 블러
      - thresh_mode: "fixed"|"mean_offset" (default "fixed")
      - offset: float (default 8)        # mean_offset 일 때 mean + offset
    """
    if img is None or img.size == 0:
        return img, {"bright_ratio": 0.0}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    k = int(params.get("blur", 0))
    if k >= 3:
        if k % 2 == 0:
            k += 1
        img8 = cv2.GaussianBlur(img8, (k, k), 0)

    base_mode = str(params.get("thresh_mode", "fixed")).lower()

    if base_mode == "mean_offset":
        offset = float(params.get("offset", 8))
        thresh = int(np.clip(float(np.mean(img8)) + offset, 0, 255))
    else:
        thresh = int(params.get("thresh", 180))

    _, bw = cv2.threshold(img8, thresh, 255, cv2.THRESH_BINARY)

    bright_ratio = float(np.count_nonzero(bw)) / float(bw.size) if bw.size else 0.0

    if bool(params.get("invert", False)):
        dbg = cv2.bitwise_not(bw)
    else:
        dbg = bw

    min_ratio = params.get("min_ratio", None)
    max_ratio = params.get("max_ratio", None)

    ok = True
    reason = "OK"

    if min_ratio is not None and bright_ratio < float(min_ratio):
        ok = False
        reason = "BRIGHT_RATIO_LOW"

    if max_ratio is not None and bright_ratio > float(max_ratio):
        ok = False
        reason = "BRIGHT_RATIO_HIGH"

    meta = {
        "bright_ratio": float(bright_ratio),
        "bright_thresh": int(thresh),
        "bright_mean": float(np.mean(img8)),
    }

    return dbg, meta, bool(ok), reason

def _presence_blob(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    """
    밝거나/어두운 blob 존재 검사

    params:
      - polarity: "bright" | "dark"      # default "bright"
      - thresh: int                      # fixed threshold
      - thresh_mode: "fixed" | "mean_offset"   # default "fixed"
      - offset: float                    # mean_offset용
      - blur: int                        # gaussian blur
      - min_area: int                    # 최소 blob 면적
      - max_area: int (optional)         # 최대 blob 면적
      - open_kernel: int (default 0)     # morphology open
      - close_kernel: int (default 0)    # morphology close
      - debug_fill: bool (default True)  # debug mask 표시
    """
    if img is None or img.size == 0:
        return img, {"blob_area": 0, "blob_count": 0}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    k = int(params.get("blur", 0))
    if k >= 3:
        if k % 2 == 0:
            k += 1
        img8 = cv2.GaussianBlur(img8, (k, k), 0)

    thresh_mode = str(params.get("thresh_mode", "fixed")).lower()
    if thresh_mode == "mean_offset":
        offset = float(params.get("offset", 8))
        thresh = int(np.clip(float(np.mean(img8)) + offset, 0, 255))
    else:
        thresh = int(params.get("thresh", 180))

    polarity = str(params.get("polarity", "bright")).lower()
    if polarity == "dark":
        _, bw = cv2.threshold(img8, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(img8, thresh, 255, cv2.THRESH_BINARY)

    open_k = int(params.get("open_kernel", 0))
    if open_k >= 2:
        kernel = np.ones((open_k, open_k), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, kernel)

    close_k = int(params.get("close_kernel", 0))
    if close_k >= 2:
        kernel = np.ones((close_k, close_k), np.uint8)
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, kernel)

    num_labels, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    min_area = int(params.get("min_area", 10))
    max_area = params.get("max_area", None)
    max_area = int(max_area) if max_area is not None else None

    best_area = 0
    blob_count = 0
    dbg = np.zeros_like(bw)
    total_area = 0

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        total_area += area

        if area < min_area:
            continue
        if max_area is not None and area > max_area:
            continue

        blob_count += 1
        if area > best_area:
            best_area = area

        if bool(params.get("debug_fill", True)):
            dbg[labels == i] = 255

    min_count = int(params.get("min_count", 1))
    min_total_area = int(params.get("min_total_area", 0))

    ok = True
    reason = "OK"

    if blob_count < min_count:
        ok = False
        reason = "BLOB_COUNT_LOW"
    elif total_area < min_total_area:
        ok = False
        reason = "BLOB_AREA_LOW"

    meta = {
        "blob_area": int(best_area),
        "blob_count": int(blob_count),
        "blob_thresh": int(thresh),
        "blob_mean": float(np.mean(img8)),
        "blob_polarity": polarity,
        "blob_total_area": int(total_area),
    }

    return dbg, meta, ok, reason

def register_measure_tools() -> None:
    register_tool("measure.edge_energy", _edge_energy)
    register_tool("measure.edge", _edge_energy)
    register_tool("measure.blob_count", _blob_count)
    register_tool("measure.dark_ratio", _dark_ratio)
    register_tool("measure.bright_ratio", _bright_ratio)
    register_tool("measure.presence_blob", _presence_blob)