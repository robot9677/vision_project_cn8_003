import cv2
import numpy as np

class ROITracker:
    def __init__(self, search_margin=20, method=cv2.TM_CCOEFF_NORMED, thr=0.6):
        self.search_margin = int(search_margin)
        self.method = method
        self.thr = float(thr)
        self.template = None  # 기준 ROI crop

    def set_template(self, tmpl_gray8: np.ndarray):
        if tmpl_gray8 is None or tmpl_gray8.size == 0:
            self.template = None
        else:
            self.template = tmpl_gray8.copy()

    def track(self, frame_gray8: np.ndarray, x, y, w, h):
        if self.template is None:
            return x, y, w, h

        H, W = frame_gray8.shape[:2]
        m = self.search_margin

        sx = max(0, x - m)
        sy = max(0, y - m)
        ex = min(W, x + w + m)
        ey = min(H, y + h + m)

        search = frame_gray8[sy:ey, sx:ex]
        th, tw = self.template.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return x, y, w, h

        res = cv2.matchTemplate(search, self.template, self.method)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)
        if maxv < self.thr:
            return x, y, w, h

        nx = sx + maxloc[0]
        ny = sy + maxloc[1]
        return nx, ny, w, h
