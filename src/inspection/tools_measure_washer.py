import cv2
import numpy as np


def run_washer_presence(crop, cfg):
    params = cfg if isinstance(cfg, dict) else {}

    blur = int(params.get("blur", 3))
    canny_low = int(params.get("canny_low", 50))
    canny_high = int(params.get("canny_high", 150))

    band_top = float(params.get("band_top", 0.3))
    band_bottom = float(params.get("band_bottom", 0.7))

    min_edge = int(params.get("min_edge", 30))

    img = crop

    if blur > 1:
        if blur % 2 == 0:
            blur += 1
        img = cv2.GaussianBlur(img, (blur, blur), 0)

    edges = cv2.Canny(img, canny_low, canny_high)

    h, w = edges.shape[:2]
    y1 = int(h * band_top)
    y2 = int(h * band_bottom)

    band = edges[y1:y2, :]

    edge_count = int(np.count_nonzero(band))

    # --- 프로파일 생성 ---
    # 중앙 60%만 사용
    h, w = img.shape[:2]
    x1 = int(w * 0.2)
    x2 = int(w * 0.8)

    profile = np.mean(img[:, x1:x2], axis=1)

    peak_count = _count_peaks(profile)

    metrics = {
        "edge_count": edge_count,
        "peak_count": peak_count,
        "profile_max": float(np.max(profile)),
        "_last_image": edges,
    }

    ok = True
    reason = "OK"
    return ok, metrics, reason

def _count_peaks(profile, min_dist=10 th_ratio=0.4):
    peaks = []
    max_val = np.max(profile)
    th = max_val * th_ratio

    for i in range(1, len(profile)-1):
        if profile[i] > profile[i-1] and profile[i] > profile[i+1] and profile[i] > th:
            if not peaks or (i - peaks[-1]) > min_dist:
                peaks.append(i)

    return len(peaks)