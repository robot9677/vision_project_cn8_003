# src/inspection/tools_measure.py
import cv2
import numpy as np
from typing import Any, Dict, Tuple
from .toolchain import register_tool

def _resolve_mm_per_px(calibration: Dict[str, Any], metrics: Dict[str, Any]) -> Tuple[Any, str]:
    if not isinstance(calibration, dict):
        return None, "NO_CALIBRATION", {}

    mode = str(calibration.get("mode", "none")).strip().lower()

    if mode in ("", "none"):
        return None, "NO_CALIBRATION", {}

    if mode == "fixed":
        mm_per_px = calibration.get("mm_per_px", None)
        if mm_per_px is None:
            return None, "CALIBRATION_MISSING_MM_PER_PX" ,{}
        return float(mm_per_px), "OK", {}

    if mode == "reference_circle":
        ref_mm = calibration.get("reference_diameter_mm", None)
        if ref_mm is None:
            return None, "CALIBRATION_MISSING_REF_MM" ,{}

        circles = metrics.get("circles") or []
        diameters_px = metrics.get("diameters_px") or []

        if not isinstance(diameters_px, list) or not diameters_px:
            return None, "CALIBRATION_REF_NOT_FOUND", {}

        selector = str(calibration.get("selector", "largest")).strip().lower()

        ref_idx = -1

        if selector == "largest":
            ref_idx = max(range(len(diameters_px)), key=lambda i: float(diameters_px[i]))
        elif selector == "smallest":
            ref_idx = min(range(len(diameters_px)), key=lambda i: float(diameters_px[i]))
        elif selector == "leftmost" and isinstance(circles, list) and len(circles) == len(diameters_px):
            ref_idx = min(
                range(len(circles)),
                key=lambda i: float(circles[i].get("x", 0)) if isinstance(circles[i], dict) else float(circles[i][0])
            )
        elif selector == "rightmost" and isinstance(circles, list) and len(circles) == len(diameters_px):
            ref_idx = max(
                range(len(circles)),
                key=lambda i: float(circles[i].get("x", 0)) if isinstance(circles[i], dict) else float(circles[i][0])
            )
        elif selector == "index":
            ref_idx = int(calibration.get("reference_index", 0))
        else:
            return None, f"CALIBRATION_BAD_SELECTOR:{selector}", {}

        if ref_idx < 0 or ref_idx >= len(diameters_px):
            return None, "CALIBRATION_REF_NOT_FOUND", {}

        ref_px = float(diameters_px[ref_idx])
        if ref_px <= 0:
            return None, "CALIBRATION_REF_INVALID", {}

        return float(ref_mm) / ref_px, "OK", {
            "selector": selector,
            "ref_index": int(ref_idx),
            "ref_px": float(ref_px)
        }

    return None, f"UNSUPPORTED_CALIBRATION_MODE:{mode}" ,{}

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

def _washer_presence(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    """
    washer 전용
    - legacy edge-band mode 지원
    - fallback blob mode 지원
    """

    if img is None or img.size == 0:
        return img, {"washer_count": 0, "edge_count": 0}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    blur = int(params.get("blur", 0))
    if blur >= 3:
        if blur % 2 == 0:
            blur += 1
        img8 = cv2.GaussianBlur(img8, (blur, blur), 0)

    mean_val = float(np.mean(img8))

    # -----------------------------
    # 1) legacy washer mode
    # -----------------------------
    use_legacy = (
        ("min_edge" in params) or
        ("band_top" in params) or
        ("band_bottom" in params) or
        ("canny_low" in params) or
        ("canny_high" in params)
    )

    if use_legacy:
        canny_low = int(params.get("canny_low", 40))
        canny_high = int(params.get("canny_high", 120))

        edges = cv2.Canny(img8, canny_low, canny_high)

        h, w = edges.shape[:2]
        band_top = float(params.get("band_top", 0.35))
        band_bottom = float(params.get("band_bottom", 0.75))

        y1 = max(0, min(h - 1, int(round(h * band_top))))
        y2 = max(y1 + 1, min(h, int(round(h * band_bottom))))

        band = edges[y1:y2, :]
        edge_count = int(np.count_nonzero(band))

        min_edge = int(params.get("min_edge", 165))
        min_mean = float(params.get("min_mean", 0.0))

        ok = (edge_count >= min_edge) and (mean_val >= min_mean)

        print(f"[DBG WASHER] edge={edge_count} mean={mean_val:.1f} min_edge={min_edge} min_mean={min_mean} ok={ok}")

        dbg = cv2.cvtColor(img8, cv2.COLOR_GRAY2BGR)
        cv2.rectangle(dbg, (0, y1), (w - 1, y2 - 1), (0, 255, 255), 1)
        dbg[y1:y2, :, 1] = np.maximum(dbg[y1:y2, :, 1], band)

        meta = {
            "washer_count": 1 if edge_count > 0 else 0,
            "edge_count": int(edge_count),
            "washer_mean": float(mean_val),
            "washer_band_top": float(band_top),
            "washer_band_bottom": float(band_bottom),
            "washer_mode": "legacy_edge_band",
        }

        if not ok:
            if edge_count < min_edge:
                return dbg, meta, False, "WASHER_EDGE_LOW"
            return dbg, meta, False, "WASHER_MEAN_LOW"

        return dbg, meta, True, "OK"

    # -----------------------------
    # 2) fallback blob mode
    # -----------------------------
    thresh_mode = str(params.get("thresh_mode", "fixed")).lower()
    if thresh_mode == "mean_offset":
        offset = float(params.get("offset", 8))
        thresh = int(np.clip(mean_val + offset, 0, 255))
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

    num, labels, stats, _ = cv2.connectedComponentsWithStats(bw, connectivity=8)

    min_area = int(params.get("min_area", 10))
    max_area = int(params.get("max_area", 999999))
    min_count = int(params.get("min_count", 1))

    count = 0
    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if min_area <= area <= max_area:
            count += 1

    ok = count >= min_count

    meta = {
        "washer_count": int(count),
        "washer_mean": float(mean_val),
        "washer_mode": "blob",
    }

    print(f"[DBG WASHER] edge={edge_count} mean={mean_val:.1f} min_edge={min_edge} min_mean={min_mean} ok={ok}")

    return bw, meta, ok, "OK" if ok else "WASHER_MISSING"

def _circle_size(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    metrics = ctx.get("metrics", {}) if isinstance(ctx, dict) else {}
    circles = metrics.get("circles") or []
    diameters_px_src = metrics.get("diameters_px") or []
    radii_px_src = metrics.get("radii_px") or []

    if not circles:
        return img, {"circle_measure_count": 0}, False, "NO_CIRCLES"

    radii_px = [float(v) for v in radii_px_src] if radii_px_src else []
    diameters_px = [float(v) for v in diameters_px_src] if diameters_px_src else []

    if not diameters_px:
        for c in circles:
            r = float(c.get("r", 0)) if isinstance(c, dict) else float(c[2])
            radii_px.append(r)
            diameters_px.append(2.0 * r)

    calibration = params.get("calibration")

    if not calibration:
        product_profile = ctx.get("product_profile") or {}
        profile_cal = product_profile.get("calibration", {}) if isinstance(product_profile, dict) else {}
        active_name = str(profile_cal.get("active", "")).strip() if isinstance(profile_cal, dict) else ""
        presets = profile_cal.get("presets", {}) if isinstance(profile_cal, dict) else {}
        calibration = presets.get(active_name, {})

    mm_per_px, calib_reason, calib_info = _resolve_mm_per_px(calibration, metrics)

    meta = {
        "circle_measure_count": int(len(circles)),

        # --- raw px ---
        "radii_px": radii_px,
        "diameters_px": diameters_px,

        # --- 통계 (px) ---
        "radius_px_avg": float(sum(radii_px) / len(radii_px)) if radii_px else 0.0,
        "radius_px_min": float(min(radii_px)) if radii_px else 0.0,
        "radius_px_max": float(max(radii_px)) if radii_px else 0.0,

        "diameter_px_avg": float(sum(diameters_px) / len(diameters_px)) if diameters_px else 0.0,
        "diameter_px_min": float(min(diameters_px)) if diameters_px else 0.0,
        "diameter_px_max": float(max(diameters_px)) if diameters_px else 0.0,

        # --- center ---
        "centers": [
            {
                "x": float(c.get("x", 0)) if isinstance(c, dict) else float(c[0]),
                "y": float(c.get("y", 0)) if isinstance(c, dict) else float(c[1]),
            }
            for c in circles
        ],

        # --- calibration ---
        "unit_mode": "px",
        "calibration_mode": str(calibration.get("mode", "none")).strip().lower() if isinstance(calibration, dict) else "none",
        "calibration_ok": bool(mm_per_px is not None),
        "calibration_reason": calib_reason,

        "calibration_selector": calib_info.get("selector"),
        "calibration_ref_index": calib_info.get("ref_index"),
        "calibration_ref_px": calib_info.get("ref_px"),
    }

    if mm_per_px is not None:
        radii_mm = [r * mm_per_px for r in radii_px]
        diameters_mm = [d * mm_per_px for d in diameters_px]

        meta.update({
            "unit_mode": "mm",
            "mm_per_px": float(mm_per_px),

            "radii_mm": radii_mm,
            "diameters_mm": diameters_mm,

            "radius_mm_avg": float(sum(radii_mm) / len(radii_mm)) if radii_mm else 0.0,
            "radius_mm_min": float(min(radii_mm)) if radii_mm else 0.0,
            "radius_mm_max": float(max(radii_mm)) if radii_mm else 0.0,

            "diameter_mm_avg": float(sum(diameters_mm) / len(diameters_mm)) if diameters_mm else 0.0,
            "diameter_mm_min": float(min(diameters_mm)) if diameters_mm else 0.0,
            "diameter_mm_max": float(max(diameters_mm)) if diameters_mm else 0.0,
        })

    if calib_info:
        print(
            f"[DBG CAL] selector={calib_info.get('selector')} "
            f"ref_idx={calib_info.get('ref_index')} "
            f"ref_px={calib_info.get('ref_px')}"
        )
    
    # --- tolerance / judge ---
    judge_mode = str(params.get("judge_mode", "roi_avg")).lower()  # roi_avg | each_circle

    target = params.get("target_mm", None)
    tol = params.get("tol_mm", None)

    if target is not None and tol is not None and mm_per_px is not None:
        target = float(target); tol = float(tol)

        # --- ROI 평균 기준 ---
        if judge_mode == "roi_avg":
            avg = meta.get("diameter_mm_avg", None)
            if avg is None:
                return img, meta, False, "NO_MEASURE"

            ok = (target - tol) <= float(avg) <= (target + tol)
            meta.update({
                "judge_mode": "roi_avg",
                "judge_value": float(avg),
                "target_mm": target,
                "tol_mm": tol,
            })
            return img, meta, bool(ok), "OK" if ok else "OUT_OF_TOL"

        # --- circle 개별 기준 ---
        elif judge_mode == "each_circle":
            vals = meta.get("diameters_mm", [])
            flags = []
            for v in vals:
                v = float(v)
                flags.append((target - tol) <= v <= (target + tol))

            ok = all(flags) if flags else False
            meta.update({
                "judge_mode": "each_circle",
                "judge_flags": flags,     # 각 circle OK/NG
                "target_mm": target,
                "tol_mm": tol,
            })
            return img, meta, bool(ok), "OK" if ok else "OUT_OF_TOL"

    return img, meta, True, "OK"

def _circle_distance(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    metrics = ctx.get("metrics", {}) if isinstance(ctx, dict) else {}

    centers = metrics.get("centers") or []
    if not centers:
        circles = metrics.get("circles") or []
        centers = [
            {
                "x": float(c.get("x", 0)) if isinstance(c, dict) else float(c[0]),
                "y": float(c.get("y", 0)) if isinstance(c, dict) else float(c[1]),
            }
            for c in circles
        ]

    points = [
        {
            "orig_index": int(i),
            "x": float(c.get("x", 0)),
            "y": float(c.get("y", 0)),
        }
        for i, c in enumerate(centers)
    ]

    order_mode = str(params.get("order_mode", "x")).strip().lower()
    if order_mode == "x":
        points = sorted(points, key=lambda p: (p["x"], p["y"]))
    elif order_mode == "y":
        points = sorted(points, key=lambda p: (p["y"], p["x"]))

    mm_per_px = metrics.get("mm_per_px", None)

    pair_mode = str(params.get("pair_mode", "all")).strip().lower()
    pair_indices = []

    if pair_mode == "adjacent":
        pair_indices = [(i, i + 1) for i in range(max(0, len(points) - 1))]
    else:
        for i in range(len(points)):
            for j in range(i + 1, len(points)):
                pair_indices.append((i, j))
    pairs = []
    dists_px = []
    dists_mm = []

    for pair_no, (i, j) in enumerate(pair_indices):
        pi = points[i]
        pj = points[j]

        xi = float(pi["x"])
        yi = float(pi["y"])
        xj = float(pj["x"])
        yj = float(pj["y"])

        dx = xi - xj
        dy = yi - yj
        dist_px = float((dx * dx + dy * dy) ** 0.5)

        item = {
            "pair_index": int(pair_no),
            "i": int(pi["orig_index"]),
            "j": int(pj["orig_index"]),
            "order_i": int(i),
            "order_j": int(j),
            "x1": float(xi),
            "y1": float(yi),
            "x2": float(xj),
            "y2": float(yj),
            "distance_px": dist_px,
        }

        dists_px.append(dist_px)

        if mm_per_px is not None:
            dist_mm = float(dist_px) * float(mm_per_px)
            item["distance_mm"] = dist_mm
            dists_mm.append(dist_mm)

        pairs.append(item)

    meta = {
        "distance_pair_count": int(len(pairs)),
        "distance_pairs": pairs,
        "center_distances_px": dists_px,
        "distance_order_mode": order_mode,
    }

    if mm_per_px is not None and dists_mm:
        meta["center_distances_mm"] = dists_mm

    judge_unit = str(
        params.get("judge_unit", "mm" if mm_per_px is not None else "px")
    ).strip().lower()
    judge_mode = str(params.get("judge_mode", "single_pair")).strip().lower()

    if judge_unit == "mm":
        target_single = params.get("target_mm", None)
        tol_single = params.get("tol_mm", None)
        target_list = params.get("target_mm_list", None)
        tol_list = params.get("tol_mm_list", None)
    else:
        target_single = params.get("target_px", None)
        tol_single = params.get("tol_px", None)
        target_list = params.get("target_px_list", None)
        tol_list = params.get("tol_px_list", None)

    if judge_mode == "each_pair":
        if not pairs:
            return img, meta, False, "NO_DISTANCE_PAIR"

        if judge_unit == "mm" and mm_per_px is None:
            return img, meta, False, "NO_SCALE"

        if target_list is None:
            if target_single is None:
                return img, meta, True, "OK"
            target_vals = [float(target_single)] * len(pairs)
        else:
            target_vals = [float(v) for v in target_list]

        if tol_list is None:
            if tol_single is None:
                return img, meta, True, "OK"
            tol_vals = [float(tol_single)] * len(pairs)
        else:
            tol_vals = [float(v) for v in tol_list]

        if len(target_vals) != len(pairs):
            return img, meta, False, "BAD_TARGET_LIST"

        if len(tol_vals) != len(pairs):
            return img, meta, False, "BAD_TOL_LIST"

        flags = []
        details = []

        for idx, pair in enumerate(pairs):
            if judge_unit == "mm":
                judge_value = float(pair.get("distance_mm", 0.0))
            else:
                judge_value = float(pair.get("distance_px", 0.0))

            target = float(target_vals[idx])
            tol = float(tol_vals[idx])
            ok = (target - tol) <= judge_value <= (target + tol)

            flags.append(bool(ok))
            details.append({
                "pair_index": int(idx),
                "i": int(pair["i"]),
                "j": int(pair["j"]),
                "x1": float(pair["x1"]),
                "y1": float(pair["y1"]),
                "x2": float(pair["x2"]),
                "y2": float(pair["y2"]),
                "unit": judge_unit,
                "value": float(judge_value),
                "target": float(target),
                "tol": float(tol),
                "ok": bool(ok),
            })

        meta.update({
            "distance_judge_mode": "each_pair",
            "distance_judge_unit": judge_unit,
            "distance_judge_flags": flags,
            "distance_judge_details": details,
        })

        if judge_unit == "mm":
            meta["distance_target_mm_list"] = target_vals
            meta["distance_tol_mm_list"] = tol_vals
        else:
            meta["distance_target_px_list"] = target_vals
            meta["distance_tol_px_list"] = tol_vals

        ok_all = all(flags) if flags else False
        return img, meta, bool(ok_all), "OK" if ok_all else "DISTANCE_OUT_OF_TOL"

    pair_index = int(params.get("pair_index", 0))

    if target_single is None or tol_single is None:
        return img, meta, True, "OK"

    if not pairs:
        return img, meta, False, "NO_DISTANCE_PAIR"

    if pair_index < 0 or pair_index >= len(pairs):
        return img, meta, False, "BAD_PAIR_INDEX"

    pair = pairs[pair_index]

    if judge_unit == "mm":
        if "distance_mm" not in pair:
            return img, meta, False, "NO_SCALE"

        judge_value = float(pair["distance_mm"])
        target = float(target_single)
        tol = float(tol_single)

        meta.update({
            "distance_judge_mode": "single_pair",
            "distance_judge_unit": "mm",
            "distance_judge_value": judge_value,
            "distance_target_mm": target,
            "distance_tol_mm": tol,
            "distance_judge_pair_index": int(pair_index),
            "distance_judge_i": int(pair["i"]),
            "distance_judge_j": int(pair["j"]),
            "distance_judge_x1": float(pair["x1"]),
            "distance_judge_y1": float(pair["y1"]),
            "distance_judge_x2": float(pair["x2"]),
            "distance_judge_y2": float(pair["y2"]),
        })
    else:
        judge_value = float(pair["distance_px"])
        target = float(target_single)
        tol = float(tol_single)

        meta.update({
            "distance_judge_mode": "single_pair",
            "distance_judge_unit": "px",
            "distance_judge_value": judge_value,
            "distance_target_px": target,
            "distance_tol_px": tol,
            "distance_judge_pair_index": int(pair_index),
            "distance_judge_i": int(pair["i"]),
            "distance_judge_j": int(pair["j"]),
            "distance_judge_x1": float(pair["x1"]),
            "distance_judge_y1": float(pair["y1"]),
            "distance_judge_x2": float(pair["x2"]),
            "distance_judge_y2": float(pair["y2"]),
        })

    ok = (target - tol) <= judge_value <= (target + tol)
    return img, meta, bool(ok), "OK" if ok else "DISTANCE_OUT_OF_TOL"

def _line_angle(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    metrics = ctx.get("metrics", {}) if isinstance(ctx, dict) else {}

    judge_mode = str(params.get("judge_mode", "global")).strip().lower()

    if judge_mode == "local":
        value = metrics.get("line_angle_local_deg", None)
    else:
        value = metrics.get("line_angle_global_deg", None)

    if value is None:
        value = metrics.get("line_angle_deg", None)

    if value is None:
        lines = metrics.get("lines") or []
        if not lines:
            return img, {"line_angle_count": 0}, False, "NO_LINE"
        ln = lines[0]
        value = float(ln.get("angle_norm", ln.get("angle", 0.0)))

    target = params.get("target_angle_deg", None)
    tol = params.get("tol_angle_deg", None)

    meta = {
        "line_angle_count": 1,
        "line_angle_judge_mode": judge_mode,
        "line_angle_value_deg": float(value),
    }

    if target is None or tol is None:
        return img, meta, True, "OK"

    target = float(target)
    tol = float(tol)

    ok = (target - tol) <= float(value) <= (target + tol)

    meta.update({
        "line_angle_target_deg": float(target),
        "line_angle_tol_deg": float(tol),
        "line_angle_ok": bool(ok),
    })

    return img, meta, bool(ok), "OK" if ok else "ANGLE_OUT_OF_TOL"

def register_measure_tools() -> None:
    register_tool("measure.edge_energy", _edge_energy)
    register_tool("measure.edge", _edge_energy)
    register_tool("measure.blob_count", _blob_count)
    register_tool("measure.dark_ratio", _dark_ratio)
    register_tool("measure.bright_ratio", _bright_ratio)
    register_tool("measure.presence_blob", _presence_blob)
    register_tool("measure.washer", _washer_presence)
    register_tool("measure.circle_size", _circle_size)
    register_tool("measure.circle_distance", _circle_distance)
    register_tool("measure.line_angle", _line_angle)