from collections import deque
import math


class Stabilizer:
    """ROI position smoothing and motion stability detection."""

    def __init__(self, window=5, move_thresh_px=3, alpha=0.6):
        self.window = int(max(1, window))
        self.thresh = float(move_thresh_px)
        self.alpha = float(alpha)
        self.state = {}

    def reset(self):
        self.state.clear()

    def _init_roi(self, roi):
        roi_id = roi.get("id")
        self.state[roi_id] = {
            "ema_x": float(roi["x"]),
            "ema_y": float(roi["y"]),
            "w": int(roi["w"]),
            "h": int(roi["h"]),
            "angle": float(roi.get("angle", 0.0)),
            "hist": deque(maxlen=self.window),
        }

    def update(self, moved_rois):
        moved_rois = list(moved_rois or [])
        if not moved_rois:
            self.reset()
            return [], False

        active_ids = {roi.get("id") for roi in moved_rois}
        for stale_id in list(self.state):
            if stale_id not in active_ids:
                del self.state[stale_id]

        for roi in moved_rois:
            roi_id = roi.get("id")
            if roi_id not in self.state:
                self._init_roi(roi)

        smoothed = []
        for roi in moved_rois:
            roi_id = roi.get("id")
            state = self.state[roi_id]
            x = float(roi["x"])
            y = float(roi["y"])

            state["ema_x"] = self.alpha * state["ema_x"] + (1.0 - self.alpha) * x
            state["ema_y"] = self.alpha * state["ema_y"] + (1.0 - self.alpha) * y
            state["w"] = int(roi.get("w", state["w"]))
            state["h"] = int(roi.get("h", state["h"]))
            state["angle"] = float(roi.get("angle", state["angle"]))

            shift = math.hypot(x - state["ema_x"], y - state["ema_y"])
            state["hist"].append(shift)

            smoothed.append({
                "id": roi_id,
                "name": roi.get("name", ""),
                "x": state["ema_x"],
                "y": state["ema_y"],
                "w": state["w"],
                "h": state["h"],
                "angle": state["angle"],
            })

        stable = all(
            len(state["hist"]) >= self.window
            and max(state["hist"]) <= self.thresh
            for state in self.state.values()
        )
        return smoothed, stable
