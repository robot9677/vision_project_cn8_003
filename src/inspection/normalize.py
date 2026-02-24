# src/inspection/normalize.py
import cv2
import numpy as np

def clahe_equalize(gray8, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray8)

def scale_to_target_mean(gray8, target_mean=120.0, max_scale=2.5, min_scale=0.5):
    m = float(max(1.0, gray8.mean()))
    scale = float(target_mean) / m
    scale = max(min_scale, min(max_scale, scale))
    out = cv2.convertScaleAbs(gray8, alpha=scale, beta=0)
    return out, scale

def normalize_frame(gray8, target_mean=120.0, do_clahe=True):
    """
    Returns (normalized_gray8, info_dict)
    info_dict: {"scale":float, "method":"scale|scale+clahe"}
    """
    if gray8 is None:
        return None, {"scale":1.0, "method":"none"}
    if gray8.dtype != 'uint8':
        gray8 = cv2.normalize(gray8, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    scaled, scale = scale_to_target_mean(gray8, target_mean=target_mean)
    method = "scale"
    if do_clahe:
        scaled = clahe_equalize(scaled)
        method = "scale+clahe"
    return scaled, {"scale": scale, "method": method}