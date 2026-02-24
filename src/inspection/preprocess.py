import cv2
import numpy as np

def normalize_by_roi(frame_u8, roi_crop, target_mean=100.0, min_gain=0.5, max_gain=6.0):
    """
    roi_crop 평균을 target_mean으로 맞추도록 frame 전체에 gain 적용
    """
    if roi_crop is None or roi_crop.size == 0:
        return frame_u8, 1.0

    mean = float(np.mean(roi_crop))
    if mean < 1e-3:
        return frame_u8, 1.0

    gain = target_mean / mean
    gain = float(np.clip(gain, min_gain, max_gain))

    out = cv2.convertScaleAbs(frame_u8, alpha=gain, beta=0)
    return out, gain
