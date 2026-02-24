# src/inspection/aligner.py
import cv2
import numpy as np

class Aligner:
    def __init__(self, template_img, search_margin=80, match_thresh=0.6):
        """
        template_img: gray8 numpy array (H,W)
        """
        if template_img is None:
            raise ValueError("template_img is None")
        self.template = template_img
        self.h, self.w = template_img.shape[:2]
        self.margin = int(search_margin)
        self.match_thresh = float(match_thresh)

        self.last_dx = 0
        self.last_dy = 0

    def update(self, frame_gray):
        """
        frame_gray: full-frame gray8 numpy array
        returns: (ok:bool, dx:int, dy:int)
        dx,dy are offsets to apply to original roi coordinates (positive means move right/down)
        """
        if frame_gray is None:
            return False, 0, 0
        H, W = frame_gray.shape[:2]

        # center guess + previous offset
        cx = W // 2 + int(self.last_dx)
        cy = H // 2 + int(self.last_dy)

        x1 = max(0, cx - self.w//2 - self.margin)
        y1 = max(0, cy - self.h//2 - self.margin)
        x2 = min(W, cx + self.w//2 + self.margin)
        y2 = min(H, cy + self.h//2 + self.margin)

        search = frame_gray[y1:y2, x1:x2]
        if search.size == 0 or search.shape[0] < self.h or search.shape[1] < self.w:
            return False, 0, 0

        res = cv2.matchTemplate(search, self.template, cv2.TM_CCOEFF_NORMED)
        minv, maxv, minp, maxp = cv2.minMaxLoc(res)

        if maxv < self.match_thresh:
            return False, 0, 0

        px = x1 + maxp[0]
        py = y1 + maxp[1]

        # Target center reference (where template would be if no shift)
        ref_x = (W//2 - self.w//2)
        ref_y = (H//2 - self.h//2)

        dx = int(px - ref_x)
        dy = int(py - ref_y)

        self.last_dx = dx
        self.last_dy = dy

        return True, dx, dy

    def apply_to_rois(self, rois, dx, dy):
        """
        rois: list of dicts {"x","y","w","h",...}
        returns: moved list with same structure
        """
        moved = []
        for r in rois:
            moved.append({
                "x": int(r["x"] + dx),
                "y": int(r["y"] + dy),
                "w": int(r["w"]),
                "h": int(r["h"]),
                "name": r.get("name", "")
            })
        return moved