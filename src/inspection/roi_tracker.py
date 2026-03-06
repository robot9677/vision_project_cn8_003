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

    def track(self, frame_gray8, x, y, w, h):

        pos = self._match(frame_gray8, x, y, w, h, self.search_margin)

        if pos is None:
            pos = self._match(frame_gray8, x, y, w, h, self.search_margin * 3)

        if pos is None:
            return x, y, w, h

        return pos
