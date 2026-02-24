# src/inspection/stabilizer.py
from collections import deque
import math
import copy

class Stabilizer:
    """
    ROI-level EMA smoothing + stability detector.

    Usage:
      st = Stabilizer(window=5, move_thresh_px=3, alpha=0.6)
      smoothed, stable = st.update(moved_rois)
    moved_rois: list of {"id","name","x","y","w","h"} (ints)
    returns:
      smoothed: same structure with float coords (x,y) smoothed
      stable: bool (True if movement below threshold for last window)
    """

    def __init__(self, window=5, move_thresh_px=3, alpha=0.6):
        self.window = int(max(1, window))
        self.thresh = float(move_thresh_px)
        self.alpha = float(alpha)
        # per-roi state: id -> {"x","y","w","h","ema_x","ema_y","hist":deque}
        self.state = {}

    def _init_roi(self, r):
        rid = r.get("id")
        self.state[rid] = {
            "ema_x": float(r["x"]),
            "ema_y": float(r["y"]),
            "w": int(r["w"]),
            "h": int(r["h"]),
            "hist": deque(maxlen=self.window)
        }

    def update(self, moved_rois):
        # moved_rois: list of dicts, preserve order
        smoothed = []
        max_shift = 0.0

        # ensure all ROI ids exist in state
        for r in moved_rois:
            rid = r.get("id")
            if rid not in self.state:
                self._init_roi(r)

        # update EMA per roi
        for r in moved_rois:
            rid = r.get("id")
            s = self.state[rid]
            x = float(r["x"]); y = float(r["y"])
            # EMA update
            s["ema_x"] = self.alpha * s["ema_x"] + (1.0 - self.alpha) * x
            s["ema_y"] = self.alpha * s["ema_y"] + (1.0 - self.alpha) * y
            s["w"] = int(r.get("w", s["w"]))
            s["h"] = int(r.get("h", s["h"]))

            # shift magnitude between current raw and ema
            dx = x - s["ema_x"]
            dy = y - s["ema_y"]
            shift = math.hypot(dx, dy)
            s["hist"].append(shift)
            if shift > max_shift:
                max_shift = shift

            smoothed.append({
                "id": rid,
                "name": r.get("name",""),
                "x": s["ema_x"],
                "y": s["ema_y"],
                "w": s["w"],
                "h": s["h"]
            })

        # stability: all ROIs have small recent shifts
        stable = True
        for rid, s in self.state.items():
            # if no recent history for this ROI, not stable
            if len(s["hist"]) < self.window:
                stable = False
                break
            # check max in deque
            if max(s["hist"]) > self.thresh:
                stable = False
                break

        return smoothed, stable