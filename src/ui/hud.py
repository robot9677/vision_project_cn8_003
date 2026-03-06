import cv2


def draw_mode_indicator(frame, edit_mode):
    text = "EDIT MODE" if edit_mode else "RUN MODE"
    color = (0, 200, 255) if edit_mode else (0, 200, 0)

    cv2.putText(
        frame,
        text,
        (20, 40),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.9,
        color,
        2,
        cv2.LINE_AA,
    )


def draw_dev_hud(frame, st):
    roi_text = getattr(st, "current_roi", "-")
    fps_val = getattr(st, "fps", 0.0)
    text = f"ROI:{roi_text}  FPS:{fps_val:.1f}"

    cv2.putText(
        frame,
        text,
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.6,
        (200, 200, 200),
        1,
        cv2.LINE_AA,
    )