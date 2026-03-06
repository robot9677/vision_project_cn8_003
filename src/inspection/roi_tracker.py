import time
import cv2
import numpy as np


class ROITracker:
    def __init__(
        self,
        search_margin=20,
        method=cv2.TM_CCOEFF_NORMED,
        thr=0.6,
        reacquire_margin=None,
        reacquire_scale=0.5,
    ):
        self.search_margin = int(search_margin)
        self.reacquire_margin = int(reacquire_margin) if reacquire_margin is not None else int(self.search_margin * 3)
        self.reacquire_scale = float(reacquire_scale)
        self.method = method
        self.thr = float(thr)
        self.template = None
        self._dbg_ts = 0.0
        self.template_alpha = 0.03
        self.update_thr = max(self.thr, 0.90)
        self.enable_template_update = False

    def set_template(self, tmpl_gray8: np.ndarray):
        if tmpl_gray8 is None or tmpl_gray8.size == 0:
            self.template = None
        else:
            self.template = tmpl_gray8.copy()
            
    def update_template(self, new_crop: np.ndarray, score: float):
        if not self.enable_template_update:
            return
        if self.template is None:
            return
        if new_crop is None or new_crop.size == 0:
            return
        if score < self.update_thr:
            return

        if new_crop.shape[:2] != self.template.shape[:2]:
            try:
                new_crop = cv2.resize(
                    new_crop,
                    (self.template.shape[1], self.template.shape[0]),
                    interpolation=cv2.INTER_AREA
                )
            except Exception:
                return

        oldf = self.template.astype(np.float32)
        newf = new_crop.astype(np.float32)

        blended = (1.0 - self.template_alpha) * oldf + self.template_alpha * newf
        self.template = np.clip(blended, 0, 255).astype(np.uint8)

    def _match_window(self, frame_gray8: np.ndarray, x, y, w, h, margin, scale=1.0):
        if self.template is None:
            return None, None

        H, W = frame_gray8.shape[:2]
        sx = max(0, int(x - margin))
        sy = max(0, int(y - margin))
        ex = min(W, int(x + w + margin))
        ey = min(H, int(y + h + margin))

        search = frame_gray8[sy:ey, sx:ex]
        th, tw = self.template.shape[:2]

        if search.shape[0] < th or search.shape[1] < tw:
            return None, None

        if 0.2 < scale < 1.0:
            sw = max(1, int(search.shape[1] * scale))
            sh = max(1, int(search.shape[0] * scale))
            tw2 = max(1, int(tw * scale))
            th2 = max(1, int(th * scale))

            search_s = cv2.resize(search, (sw, sh), interpolation=cv2.INTER_AREA)
            tmpl_s = cv2.resize(self.template, (tw2, th2), interpolation=cv2.INTER_AREA)

            if search_s.shape[0] < tmpl_s.shape[0] or search_s.shape[1] < tmpl_s.shape[1]:
                return None, None

            res = cv2.matchTemplate(search_s, tmpl_s, self.method)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)

            nx = sx + int(round(maxloc[0] / scale))
            ny = sy + int(round(maxloc[1] / scale))
            return (nx, ny, w, h), float(maxv)

        res = cv2.matchTemplate(search, self.template, self.method)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)

        nx = sx + maxloc[0]
        ny = sy + maxloc[1]
        return (nx, ny, w, h), float(maxv)

    def track(self, frame_gray8: np.ndarray, x, y, w, h):
        if self.template is None:
            return x, y, w, h

        # 1차: 근거리 탐색
        pos1, score1 = self._match_window(
            frame_gray8, x, y, w, h,
            margin=self.search_margin,
            scale=1.0,
        )

        if pos1 is not None and score1 is not None and score1 >= self.thr:
            return pos1

        # 2차: 재획득 탐색
        pos2, score2 = self._match_window(
            frame_gray8, x, y, w, h,
            margin=self.reacquire_margin,
            scale=self.reacquire_scale,
        )

        if pos2 is not None and score2 is not None and score2 >= self.thr:
            rx, ry, _, _ = pos2

            # 재획득 후보 정밀 검증
            pos3, score3 = self._match_window(
                frame_gray8, rx, ry, w, h,
                margin=self.search_margin,
                scale=1.0,
            )

            if pos3 is not None and score3 is not None and score3 >= max(self.thr, 0.72):
                now = time.time()
                if now - self._dbg_ts > 1.0:
                    print(f"[TRK] reacquired score={score2:.3f} verify={score3:.3f}")
                    self._dbg_ts = now
                return pos3

            return x, y, w, h

        return x, y, w, h