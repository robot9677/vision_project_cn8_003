import cv2


def draw_pose_message(frame, pose_bad_cnt, threshold):
    if pose_bad_cnt < threshold:
        return

    cv2.putText(
        frame,
        "POSE NOT STABLE",
        (400, 50),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.8,
        (0, 0, 255),
        2,
        cv2.LINE_AA,
    )