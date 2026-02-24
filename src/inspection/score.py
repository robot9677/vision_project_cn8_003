# src/inspection/score.py
import cv2
import numpy as np

def edge_density_score(crop, canny_thresh1=50, canny_thresh2=150):
    if crop is None or crop.size == 0:
        return 0.0
    edges = cv2.Canny(crop, canny_thresh1, canny_thresh2)
    den = float(edges.sum()) / (255.0 * edges.size)
    return min(1.0, den * 5.0)

def texture_score(crop):
    if crop is None or crop.size == 0:
        return 0.0
    s = float(crop.std()) / 64.0
    return min(1.0, s)

def combined_score(crop, w_edge=0.6, w_texture=0.4):
    e = edge_density_score(crop)
    t = texture_score(crop)
    score = w_edge * e + w_texture * t
    return float(score)