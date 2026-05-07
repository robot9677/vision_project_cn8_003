import math
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

def _circle_arc_coverage(edge_img, cx, cy, r, tol=2):
    pts = 360
    hit = 0

    for deg in range(pts):
        theta = np.deg2rad(deg)

        found_edge = False
        for dt in range(-tol, tol + 1):
            rr = r + dt
            x = int(round(cx + rr * np.cos(theta)))
            y = int(round(cy + rr * np.sin(theta)))

            if 0 <= y < edge_img.shape[0] and 0 <= x < edge_img.shape[1]:
                if edge_img[y, x] > 0:
                    found_edge = True
                    break

        if found_edge:
            hit += 1

    return float(hit) / float(pts)

def _find_circle(crop, params, ctx):
    if crop is None or crop.size == 0:
        return crop, {"circle_count": 0, "circles": []}, False, "EMPTY_CROP"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        
    gray = cv2.equalizeHist(gray)

    blur_k = int(params.get("blur", 5))
    if blur_k < 1:
        blur_k = 1
    if blur_k % 2 == 0:
        blur_k += 1

    gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)
    edges = cv2.Canny(gray_blur, 50, 150)
    gray_blur = edges

    coverage_min = float(params.get("coverage_min", 0.70))

    circles = cv2.HoughCircles(
        gray_blur,
        cv2.HOUGH_GRADIENT,
        dp=float(params.get("dp", 1.2)),
        minDist=float(params.get("min_dist", 20)),
        param1=float(params.get("param1", 100)),
        param2=float(params.get("param2", 20)),
        minRadius=int(params.get("min_radius", 0)),
        maxRadius=int(params.get("max_radius", 0)),
    )

    found = []
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if circles is not None:
        circles = np.round(circles[0]).astype(int)
        for x, y, r in circles:
            coverage = _circle_arc_coverage(edges, int(x), int(y), int(r), tol=2)
            if coverage < coverage_min:
                continue

            found.append({
                "x": int(x),
                "y": int(y),
                "r": int(r),
                "coverage": float(coverage),
            })

            cv2.circle(dbg, (int(x), int(y)), int(r), (0, 255, 0), 2)
            cv2.circle(dbg, (int(x), int(y)), 2, (0, 0, 255), -1)

        # --- circle 정렬 (좌→우 기준) ---
        found = sorted(found, key=lambda c: c["x"])

        smooth = bool(params.get("smooth_radius", False))
        alpha = float(params.get("smooth_alpha", 0.6))
        alpha = max(0.0, min(1.0, alpha))

        prev_circles = ctx.get("_prev_circles", None)

        if smooth and isinstance(prev_circles, list) and len(prev_circles) == len(found):
            for i, c in enumerate(found):
                prev_c = prev_circles[i]

                prev_x = float(prev_c.get("x", c["x"]))
                prev_y = float(prev_c.get("y", c["y"]))
                prev_r = float(prev_c.get("r", c["r"]))

                cur_x = float(c["x"])
                cur_y = float(c["y"])
                cur_r = float(c["r"])

                c["x"] = int(round(alpha * cur_x + (1.0 - alpha) * prev_x))
                c["y"] = int(round(alpha * cur_y + (1.0 - alpha) * prev_y))
                c["r"] = int(round(alpha * cur_r + (1.0 - alpha) * prev_r))

    count = len(found)
    expected = params.get("expected")
    min_count = params.get("min_count")
    max_count = params.get("max_count")

    ok = True
    reason = "OK"
    if expected is not None and count != int(expected):
        ok = False
        reason = "CIRCLE_COUNT_MISMATCH"
    if min_count is not None and count < int(min_count):
        ok = False
        reason = "CIRCLE_COUNT_LOW"
    if max_count is not None and count > int(max_count):
        ok = False
        reason = "CIRCLE_COUNT_HIGH"

    # 평균 반지름 계산
    avg_radius = 0.0
    if found:
        rs = [c["r"] if isinstance(c, dict) else c[2] for c in found]
        avg_radius = float(sum(rs) / len(rs))

    diameters_px = []
    radii_px = []

    for c in found:
        r = c["r"] if isinstance(c, dict) else c[2]
        radii_px.append(float(r))
        diameters_px.append(float(2.0 * r))

    meta = {
        "circle_count": int(count),
        "circles": found,
        "blob": int(count),
        "avg_radius": avg_radius,
        "radii_px": radii_px,
        "diameters_px": diameters_px,
    }
    ctx["_prev_circles"] = [
        {"x": int(c["x"]), "y": int(c["y"]), "r": int(c["r"])}
        for c in found
    ]

    print(
        f"[DBG CIRCLE] count={count} circles={found} "
        f"blob={count} avg_radius={avg_radius} diameters_px={diameters_px}"
    )

    return dbg, meta, bool(ok), reason


def _find_line(crop, params, ctx):
    if crop is None or crop.size == 0:
        return crop, {"line_count": 0, "lines": []}, False, "EMPTY_CROP"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur_k = int(params.get("blur", 3))
    if blur_k < 1:
        blur_k = 1
    if blur_k % 2 == 0:
        blur_k += 1
    gray_blur = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    c1 = int(params.get("canny1", 50))
    c2 = int(params.get("canny2", 150))
    edges = cv2.Canny(gray_blur, c1, c2)

    raw_lines = cv2.HoughLinesP(
        edges,
        rho=float(params.get("rho", 1)),
        theta=float(params.get("theta_deg", 1.0)) * np.pi / 180.0,
        threshold=int(params.get("threshold", 40)),
        minLineLength=int(params.get("min_length", 30)),
        maxLineGap=int(params.get("max_gap", 10)),
    )

    found = []
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    if raw_lines is not None:
        for row in raw_lines:
            x1, y1, x2, y2 = [int(v) for v in row[0]]
            dx = x2 - x1
            dy = y2 - y1
            angle = math.degrees(math.atan2(dy, dx))
            length = float(math.hypot(dx, dy))

            found.append({
                "x1": x1, "y1": y1,
                "x2": x2, "y2": y2,
                "angle": float(angle),
                "length": length,
            })
            cv2.line(dbg, (x1, y1), (x2, y2), (0, 255, 0), 2)

    best = None
    if found:
        found = sorted(found, key=lambda v: v["length"], reverse=True)
        best = found[0]

        ang = float(best["angle"])
        if ang > 90.0:
            ang -= 180.0
        elif ang <= -90.0:
            ang += 180.0

        best["angle_norm"] = float(ang)
        best["cx"] = float((best["x1"] + best["x2"]) / 2.0)
        best["cy"] = float((best["y1"] + best["y2"]) / 2.0)

    count = len(found)
    expected = params.get("expected")
    min_count = params.get("min_count")
    max_count = params.get("max_count")

    ok = True
    reason = "OK"
    if expected is not None and count != int(expected):
        ok = False
        reason = "LINE_COUNT_MISMATCH"
    if min_count is not None and count < int(min_count):
        ok = False
        reason = "LINE_COUNT_LOW"
    if max_count is not None and count > int(max_count):
        ok = False
        reason = "LINE_COUNT_HIGH"

    meta = {
        "line_count": int(count),
        "lines": found,
    }

    if best is not None:
        meta["line_p1"] = [int(best["x1"]), int(best["y1"])]
        meta["line_p2"] = [int(best["x2"]), int(best["y2"])]
        meta["line_center"] = [float(best["cx"]), float(best["cy"])]
        meta["line_angle_deg"] = float(best.get("angle_norm", best["angle"]))
        meta["line_length"] = float(best["length"])

    return dbg, meta, bool(ok), reason


def _find_blob_center(crop, params, ctx):
    if crop is None or crop.size == 0:
        return crop, {"blob_count": 0, "blob_centers": []}, False, "EMPTY_CROP"

    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY) if crop.ndim == 3 else crop
    if gray.dtype != np.uint8:
        gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    blur_k = int(params.get("blur", 0))
    if blur_k >= 3:
        if blur_k % 2 == 0:
            blur_k += 1
        gray = cv2.GaussianBlur(gray, (blur_k, blur_k), 0)

    thresh_mode = str(params.get("thresh_mode", "fixed")).lower()
    if thresh_mode == "mean_offset":
        offset = float(params.get("offset", 0))
        thresh = int(np.clip(float(np.mean(gray)) + offset, 0, 255))
    else:
        thresh = int(params.get("thresh", 127))

    polarity = str(params.get("polarity", "bright")).lower()
    if polarity == "dark":
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY_INV)
    else:
        _, bw = cv2.threshold(gray, thresh, 255, cv2.THRESH_BINARY)

    open_k = int(params.get("open", 0))
    if open_k >= 2:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (open_k, open_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, ker)

    close_k = int(params.get("close", 0))
    if close_k >= 2:
        ker = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (close_k, close_k))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, ker)

    num, labels, stats, centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    area_min = int(params.get("area_min", 0))
    area_max = int(params.get("area_max", 999999999))

    centers = []
    dbg = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    for i in range(1, num):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if not (area_min <= area <= area_max):
            continue

        x = int(round(centroids[i][0]))
        y = int(round(centroids[i][1]))
        w = int(stats[i, cv2.CC_STAT_WIDTH])
        h = int(stats[i, cv2.CC_STAT_HEIGHT])
        left = int(stats[i, cv2.CC_STAT_LEFT])
        top = int(stats[i, cv2.CC_STAT_TOP])

        centers.append({
            "x": x,
            "y": y,
            "area": area,
            "box": [left, top, w, h],
        })

        cv2.rectangle(dbg, (left, top), (left + w, top + h), (0, 255, 0), 1)
        cv2.circle(dbg, (x, y), 3, (0, 0, 255), -1)

    count = len(centers)
    expected = params.get("expected")
    min_count = params.get("min_count")
    max_count = params.get("max_count")

    ok = True
    reason = "OK"
    if expected is not None and count != int(expected):
        ok = False
        reason = "BLOB_COUNT_MISMATCH"
    if min_count is not None and count < int(min_count):
        ok = False
        reason = "BLOB_COUNT_LOW"
    if max_count is not None and count > int(max_count):
        ok = False
        reason = "BLOB_COUNT_HIGH"

    meta = {
        "blob_count": int(count),
        "blob_centers": centers,
        "blob_thresh": int(thresh),
        "blob_polarity": polarity,
    }
    return dbg, meta, bool(ok), reason

def register_locate_tools() -> None:
    register_tool("locate.pattern_match", _pattern_match)
    register_tool("locate.match", _pattern_match)
    register_tool("locate.circle", _find_circle)
    register_tool("locate.line", _find_line)
    register_tool("locate.blob_center", _find_blob_center)