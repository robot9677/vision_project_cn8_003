# src/inspection/logger.py
import os, time, cv2

def save_snapshot(root, frame_gray8, roi, prefix="stable"):
    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    rid = roi.get("id", "r")
    fn_crop = os.path.join(root, f"{prefix}_roi{rid}_{ts}.png")
    x,y,w,h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
    crop = frame_gray8[y:y+h, x:x+w]
    cv2.imwrite(fn_crop, crop)
    return fn_crop

def save_template_copy(root, tpl_img):
    os.makedirs(root, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    fn = os.path.join(root, f"template_{ts}.png")
    cv2.imwrite(fn, tpl_img)
    return fn