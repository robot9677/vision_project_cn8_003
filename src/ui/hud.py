import cv2
from ui import overlay_clean as overlay

DEV_MODE = True


def draw_dev_hud(img, st, product_profile=None):
    if product_profile is not None:
        if not product_profile.get("ui", {}).get("show_dev_hud", True):
            return
    if not DEV_MODE:
        return

    h, w = img.shape[:2]
    if st.edit_mode:
        text1 = "EDIT: n=next  x=delete  r=clear  s=save  e=run"
    else:
        text1 = "RUN: SPACE=inspect  a=autoInspect  c=autotune  p=reload  e=edit"

    ovl = img.copy()
    overlay.draw_rect(ovl, (8, h - 64), (w - 8, h - 8), color=(0, 0, 0), fill=True)
    cv2.addWeighted(ovl, 0.45, img, 0.55, 0, img)
    overlay.draw_text(img, text1, (16, h - 80), color=(220, 220, 220), scale=0.6, thickness=1, align="lt")

    if not st.edit_mode:
        hint = "sample img [ T=temp  K:OK_S  N:NG_S ]"
        x = 16
        y = h - 100
        ovl2 = img.copy()
        cv2.addWeighted(ovl2, 0.45, img, 0.55, 0, img)
        overlay.draw_text(img, hint, (x, y), color=(220, 220, 220), scale=0.55, thickness=1, align="lt")


def draw_mode_indicator(img, edit_mode):
    h, w = img.shape[:2]
    text = "EDIT MODE" if edit_mode else "RUN MODE"
    color = (0, 200, 255) if edit_mode else (0, 200, 0)

    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = w - tw - 16
    y = 28

    ovl = img.copy()
    overlay.draw_rect(ovl, (x - 8, y - 22), (x + tw + 8, y + 6), color=(0, 0, 0), fill=True)
    cv2.addWeighted(ovl, 0.4, img, 0.6, 0, img)
    overlay.draw_text(img, text, (x, y), color=color, scale=0.7, thickness=2, align="lt")

    if DEV_MODE:
        overlay.draw_text(img, "DEV", (x, y + 20), color=(180, 180, 180), scale=0.5, thickness=1, align="lt")