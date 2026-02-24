import json
import os
import cv2

class ROIManager:
    """
    ROI 포맷(통일):
      {"id": int, "name": str, "x": int, "y": int, "w": int, "h": int}
    """
    def __init__(self, frame_size):
        self.W, self.H = frame_size
        self.rois = []
        self.selected_id = None
        self._next_id = 1

    def _clamp_rect(self, x, y, w, h):
        x = max(0, min(int(x), self.W - 1))
        y = max(0, min(int(y), self.H - 1))
        w = max(1, int(w))
        h = max(1, int(h))
        if x + w > self.W:
            w = self.W - x
        if y + h > self.H:
            h = self.H - y
        return x, y, w, h

    def add(self, x, y, w, h, name=None):
        x, y, w, h = self._clamp_rect(x, y, w, h)
        roi = {
            "id": self._next_id,
            "name": name or f"ROI{self._next_id}",
            "x": int(x), "y": int(y), "w": int(w), "h": int(h)
        }
        self.rois.append(roi)
        self.selected_id = roi["id"]
        self._next_id += 1
        return roi["id"]

    def remove(self, roi_id):
        before = len(self.rois)
        self.rois = [r for r in self.rois if r.get("id") != roi_id]
        if not self.rois:
            self.selected_id = None
        else:
            if self.selected_id == roi_id:
                self.selected_id = self.rois[0]["id"]
        return len(self.rois) < before

    def get(self, roi_id):
        for r in self.rois:
            if r.get("id") == roi_id:
                return r
        return None

    def update(self, roi_id, x=None, y=None, w=None, h=None, name=None):
        r = self.get(roi_id)
        if not r:
            return False
        nx = x if x is not None else r["x"]
        ny = y if y is not None else r["y"]
        nw = w if w is not None else r["w"]
        nh = h if h is not None else r["h"]
        nx, ny, nw, nh = self._clamp_rect(nx, ny, nw, nh)
        r.update({"x": int(nx), "y": int(ny), "w": int(nw), "h": int(nh)})
        if name is not None:
            r["name"] = name
        return True

    def list(self):
        return list(self.rois)

    def select(self, roi_id):
        if any(r.get("id") == roi_id for r in self.rois):
            self.selected_id = roi_id
            return True
        return False

    def to_dict(self):
        return {"frame_size": [self.W, self.H], "rois": list(self.rois)}

    def save(self, filepath):
        d = self.to_dict()
        d["_next_id"] = self._next_id
        try:
            with open(filepath, "w", encoding="utf-8") as f:
                json.dump(d, f, indent=2, ensure_ascii=False)
            return True
        except Exception:
            return False

    def load(self, filepath):
        if not os.path.exists(filepath):
            return False

        try:
            with open(filepath, "r", encoding="utf-8") as f:
                d = json.load(f)
        except Exception:
            return False

        # ---- frame_size 호환 처리 ----
        fs = d.get("frame_size")

        if isinstance(fs, dict):
            fw = fs.get("width") or fs.get("w")
            fh = fs.get("height") or fs.get("h")
            if fw is not None and fh is not None:
                fs = [fw, fh]

        if isinstance(fs, (list, tuple)) and len(fs) >= 2:
            if fs[0] != self.W or fs[1] != self.H:
                print(f"[ROI] frame_size mismatch: file={fs}, current=({self.W},{self.H})")
        # -----------------------------

        self.rois = []
        max_id = 0
        for r in d.get("rois", []):
            roi_id = int(r.get("id", self._next_id))
            x = int(r.get("x", 0))
            y = int(r.get("y", 0))
            w = int(r.get("w", 1))
            h = int(r.get("h", 1))
            x, y, w, h = self._clamp_rect(x, y, w, h)
            self.rois.append({
                "id": roi_id,
                "name": r.get("name", f"ROI{roi_id}"),
                "x": x, "y": y, "w": w, "h": h
            })
            max_id = max(max_id, roi_id)
        self._next_id = max_id + 1 if max_id > 0 else self._next_id
        self.selected_id = self.rois[0]["id"] if self.rois else None
        return True

    # -----------------------
    # convenience helpers
    # -----------------------
    def get_selected(self):
        """Return the currently selected ROI dict or None."""
        if self.selected_id is None:
            return None
        return self.get(self.selected_id)

    def select_next(self):
        """Select next ROI in list (wrap-around). Returns new selected_id or None."""
        if not self.rois:
            self.selected_id = None
            return None
        ids = [r["id"] for r in self.rois]
        if self.selected_id is None:
            self.selected_id = ids[0]
            return self.selected_id
        try:
            idx = ids.index(self.selected_id)
            idx = (idx + 1) % len(ids)
            self.selected_id = ids[idx]
            return self.selected_id
        except ValueError:
            # current selected not found -> pick first
            self.selected_id = ids[0]
            return self.selected_id

    def delete_selected(self):
        """Delete currently selected ROI. Returns True if deleted, False otherwise."""
        if self.selected_id is None:
            return False
        rid = self.selected_id
        ok = self.remove(rid)
        # remove() already adjusts selected_id; ensure it if empty
        if not self.rois:
            self.selected_id = None
        return ok

    def clear(self):
        """Remove all ROIs (in-memory) - does not auto-save."""
        self.rois = []
        self.selected_id = None
        self._next_id = 1

    def crop(self, frame, roi_id):
        """
        Return cropped image from frame for roi_id.
        Frame is expected as numpy array (H,W) or (H,W,3).
        Returns None if ROI or frame invalid.
        """
        r = self.get(roi_id)
        if r is None:
            return None
        x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
        # clamp to frame
        Hf, Wf = frame.shape[:2]
        x1 = max(0, min(x, Wf - 1))
        y1 = max(0, min(y, Hf - 1))
        x2 = max(0, min(x + w, Wf))
        y2 = max(0, min(y + h, Hf))
        if x2 <= x1 or y2 <= y1:
            return None
        return frame[y1:y2, x1:x2].copy()

    def save_alignment_template(self, frame_gray8, filepath, roi_id=None):
        """
        Save a gray8 crop of roi_id (or selected ROI if roi_id is None) to filepath (png).
        frame_gray8: numpy ndarray (8-bit gray)
        filepath: full path string to save png
        returns: True on success, False otherwise
        """
        if frame_gray8 is None:
            return False
        # choose roi
        if roi_id is None:
            roi = self.get_selected()
        else:
            roi = self.get(roi_id)
        if roi is None:
            return False

        x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
        Hf, Wf = frame_gray8.shape[:2]
        # clamp
        x = max(0, min(x, Wf - 1))
        y = max(0, min(y, Hf - 1))
        w = max(1, min(w, Wf - x))
        h = max(1, min(h, Hf - y))
        crop = frame_gray8[y:y+h, x:x+w]
        if crop.size == 0:
            return False

        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        # write as 8-bit PNG
        import cv2
        cv2.imwrite(filepath, crop)
        return True

    def get_rois(self):
        """
        Compatibility helper: return list of ROI dicts.
        Tries several common internal names / methods.
        """
        # common attribute names
        if hasattr(self, "rois"):
            return getattr(self, "rois")
        if hasattr(self, "_rois"):
            return getattr(self, "_rois")
        # common getter methods
        if hasattr(self, "list_rois"):
            try:
                return self.list_rois()
            except Exception:
                pass
        if hasattr(self, "get_all"):
            try:
                return self.get_all()
            except Exception:
                pass
        # best-effort: try existing get() calls to build list
        try:
            # if there's a length or keys
            idx = 0
            res = []
            while True:
                r = self.get(idx)
                if r is None:
                    break
                res.append(r)
                idx += 1
        except Exception:
            pass
        # fallback: empty list
        return []
