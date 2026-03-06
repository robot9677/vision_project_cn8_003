import cv2
import numpy as np

class ROITracker:
    def __init__(self, search_margin=20, method=cv2.TM_CCOEFF_NORMED, thr=0.6,
                 reacquire_margin=None, reacquire_scale=0.5):
        self.search_margin = int(search_margin)
        self.reacquire_margin = int(reacquire_margin) if reacquire_margin is not None else int(self.search_margin * 4)
        self.reacquire_scale = float(reacquire_scale)
        self.method = method
        self.thr = float(thr)
        self.template = None

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
        res = cv2.matchTemplate(search, self.template, self.method)
        _, maxv, _, maxloc = cv2.minMaxLoc(res)

        if maxv < self.thr:
            print("[TRK] local fail -> reacquire")
            # ---- reacquire 1-shot (bigger window + optional downsample) ----
            m2 = self.reacquire_margin
            sx2 = max(0, x - m2)
            sy2 = max(0, y - m2)
            ex2 = min(W, x + w + m2)
            ey2 = min(H, y + h + m2)

            search2 = frame_gray8[sy2:ey2, sx2:ex2]
            th, tw = self.template.shape[:2]
            if search2.shape[0] < th or search2.shape[1] < tw:
                return x, y, w, h

            sc = self.reacquire_scale
            if 0.2 < sc < 1.0:
                # downsample both for speed
                sw = max(1, int(search2.shape[1] * sc))
                sh = max(1, int(search2.shape[0] * sc))
                tw2 = max(1, int(tw * sc))
                th2 = max(1, int(th * sc))

                search2s = cv2.resize(search2, (sw, sh), interpolation=cv2.INTER_AREA)
                tmpls = cv2.resize(self.template, (tw2, th2), interpolation=cv2.INTER_AREA)

                if search2s.shape[0] < tmpls.shape[0] or search2s.shape[1] < tmpls.shape[1]:
                    return x, y, w, h

                res2 = cv2.matchTemplate(search2s, tmpls, self.method)
                _, maxv2, _, maxloc2 = cv2.minMaxLoc(res2)

                if maxv2 < self.thr:
                    return x, y, w, h

                nx = sx2 + int(round(maxloc2[0] / sc))
                ny = sy2 + int(round(maxloc2[1] / sc))
                return nx, ny, w, h

            else:
                # full-res reacquire
                res2 = cv2.matchTemplate(search2, self.template, self.method)
                _, maxv2, _, maxloc2 = cv2.minMaxLoc(res2)

                if maxv2 < self.thr:
                    return x, y, w, h

                nx = sx2 + maxloc2[0]
                ny = sy2 + maxloc2[1]
                return nx, ny, w, h

        nx = sx + maxloc[0]
        ny = sy + maxloc[1]
        return nx, ny, w, h
