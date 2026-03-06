import cv2
from ui import overlay_clean as overlay


def draw_pose_message(img, pose_bad_cnt, threshold):
    if pose_bad_cnt < threshold:
        return img

    h, w = img.shape[:2]
    x = 30
    y = 80
    msg = "정면으로 맞춰주세요 (±10~15°)"

    ovl = img.copy()
    overlay.draw_rect(ovl, (x - 12, y - 30), (w - 30, y + 10), color=(0, 0, 0), fill=True)
    cv2.addWeighted(ovl, 0.45, img, 0.55, 0, img)

    try:
        img = overlay.draw_text_kr(img, msg, (x, y - 20))
    except Exception:
        overlay.draw_text(img, "Align front (±10~15 deg)", (x, y - 20), color=(255, 255, 255), scale=0.8, thickness=2, align="lt")

    return img