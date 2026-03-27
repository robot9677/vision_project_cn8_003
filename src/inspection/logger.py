# src/inspection/logger.py
import os, time, cv2

def save_snapshot(root, frame_gray8, roi, prefix="stable"):
    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rid = roi.get("id", "r")
    fn_crop = os.path.join(root, f"{prefix}_roi{rid}_{ts}.png")

    H, W = frame_gray8.shape[:2]
    x = int(roi["x"])
    y = int(roi["y"])
    w = int(roi["w"])
    h = int(roi["h"])

    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(W, x1 + w))
    y2 = max(y1 + 1, min(H, y1 + h))

    crop = frame_gray8[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None

    cv2.imwrite(fn_crop, crop)
    return fn_crop

def save_template_copy(root, tpl_img):
    if tpl_img is None or tpl_img.size == 0:
        return None

    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(root, f"template_{ts}.png")
    cv2.imwrite(fn, tpl_img)
    return fn

def prune_log_dirs(log_root, keep=10):
    try:
        days = []
        for d in os.listdir(log_root):
            p = os.path.join(log_root, d)
            if os.path.isdir(p):
                days.append((os.path.getmtime(p), p))
        days.sort(reverse=True)

        for _, dpath in days:
            sub = []
            for s in os.listdir(dpath):
                sp = os.path.join(dpath, s)
                if os.path.isdir(sp):
                    sub.append((os.path.getmtime(sp), sp))
            sub.sort(reverse=True)

            for _, sp in sub[keep:]:
                try:
                    import shutil
                    shutil.rmtree(sp)
                except:
                    pass
    except:
        pass