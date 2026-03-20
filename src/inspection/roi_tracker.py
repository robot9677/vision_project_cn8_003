import time
import cv2
import numpy as np


class ROITracker:
    def __init__(
        self,
        search_margin=20,
        search_margin_x=None,
        search_margin_y=None,
        method=cv2.TM_CCOEFF_NORMED,
        thr=0.6,
        reacquire_margin=None,
        reacquire_scale=0.5,
    ):
        self.search_margin = int(search_margin)
        self.search_margin_x = int(search_margin_x) if search_margin_x is not None else int(search_margin)
        self.search_margin_y = int(search_margin_y) if search_margin_y is not None else int(search_margin)

        self.reacquire_margin = int(reacquire_margin) if reacquire_margin is not None else int(self.search_margin * 3)
        self.reacquire_scale = float(reacquire_scale)
        self.method = method
        self.thr = float(thr)
        self.template = None
        self._dbg_ts = 0.0
        self.template_alpha = 0.03
        self.update_thr = max(self.thr, 0.90)
        self.enable_template_update = False
        self.wide_reacquire_margin_x = int(self.search_margin_x * 3)
        self.wide_reacquire_margin_y = int(self.search_margin_y * 3)
        self.wide_reacquire_scale = 0.4
        self.wide_reacquire_thr = max(0.50, self.thr - 0.1)

    def _prep_track_img(self, img):
        if img is None or img.size == 0:
            return img
        if len(img.shape) == 3:
            img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        img = clahe.apply(img)
        return img
            
    def set_template(self, tmpl_gray8: np.ndarray):
        if tmpl_gray8 is None or tmpl_gray8.size == 0:
            self.template = None
        else:
            self.template = self._prep_track_img(tmpl_gray8.copy())
            
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

        if isinstance(margin, (tuple, list)) and len(margin) >= 2:
            mx = int(margin[0])
            my = int(margin[1])
        else:
            mx = int(margin)
            my = int(margin)

        H, W = frame_gray8.shape[:2]
        sx = max(0, int(x - mx))
        sy = max(0, int(y - my))
        ex = min(W, int(x + w + mx))
        ey = min(H, int(y + h + my))

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

        pos1, score1 = self._match_window(
            frame_gray8, x, y, w, h,
            margin=(self.search_margin_x, self.search_margin_y),
            scale=1.0,
        )

        if pos1 is not None and score1 is not None and score1 >= self.thr:
            return pos1

        pos2, score2 = self._match_window(
            frame_gray8, x, y, w, h,
            margin=(self.reacquire_margin, self.reacquire_margin),
            scale=self.reacquire_scale,
        )

        if pos2 is not None and score2 is not None and score2 >= self.thr:
            rx, ry, _, _ = pos2

            pos3, score3 = self._match_window(
                frame_gray8, rx, ry, w, h,
                margin=(self.search_margin_x, self.search_margin_y),
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
    
    def _rotate_keep_size(self, img: np.ndarray, angle: float) -> np.ndarray:
        h, w = img.shape[:2]
        M = cv2.getRotationMatrix2D((w / 2.0, h / 2.0), angle, 1.0)
        return cv2.warpAffine(
            img,
            M,
            (w, h),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

    def track_pose(
        self,
        frame_gray8: np.ndarray,
        x, y, w, h,
        angle=0.0,
        angle_range=0.0,
        angle_step=1.0,
        base_angle=0.0,
        enable_rotation=False,
        max_abs_angle=8.0,
    ):

        maxv = None
        score_wide = None
        score_refine = None
        dbg_stage = "init"

        if self.template is None:
            return x, y, w, h, float(base_angle), 0.0
        
        frame_gray8 = self._prep_track_img(frame_gray8)

        H, W = frame_gray8.shape[:2]
        sx = max(0, int(x - self.search_margin_x))
        sy = max(0, int(y - self.search_margin_y))
        ex = min(W, int(x + w + self.search_margin_x))
        ey = min(H, int(y + h + self.search_margin_y))
        search = frame_gray8[sy:ey, sx:ex]

        if search.size == 0:
            return x, y, w, h, float(base_angle), 0.0

        th, tw = self.template.shape[:2]
        if search.shape[0] < th or search.shape[1] < tw:
            return x, y, w, h, float(base_angle), 0.0

        if not enable_rotation:
            res = cv2.matchTemplate(search, self.template, self.method)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)

            tracker_cfg = {}
            use_fb = True
            fb_thr = 0.62
            if use_fb and maxv < self.thr:
                def _grad(img):
                    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    return mag.astype("uint8")

                search_g = _grad(frame_gray8)
                templ_g = _grad(self.template)

                res_g = cv2.matchTemplate(search_g, templ_g, cv2.TM_CCOEFF_NORMED)
                _, score_g, _, loc_g = cv2.minMaxLoc(res_g)

                if score_g >= fb_thr:
                    nxg, nyg = loc_g
                    return nxg, nyg, w, h, float(base_angle), float(score_g)
                
            nx = sx + int(maxloc[0])
            ny = sy + int(maxloc[1])

            print(f"[DBG TRK] local score={float(maxv):.3f} thr={float(self.thr):.3f} pos=({nx},{ny})")
            if float(maxv) >= self.thr:
                return nx, ny, w, h, float(base_angle), float(maxv)

            pos_wide, score_wide = self._match_window(
                frame_gray8,
                x, y, w, h,
                margin=(self.wide_reacquire_margin_x, self.wide_reacquire_margin_y),
                scale=self.wide_reacquire_scale,
            )

            if pos_wide is not None and score_wide is not None and score_wide >= self.wide_reacquire_thr:
                rx, ry, _, _ = pos_wide
                pos_refine, score_refine = self._match_window(
                    frame_gray8,
                    rx, ry, w, h,
                    margin=(self.search_margin_x, self.search_margin_y),
                    scale=1.0,
                )

                if pos_refine is not None and score_refine is not None and score_refine >= self.thr:
                    nx2, ny2, _, _ = pos_refine
                    return nx2, ny2, w, h, float(base_angle), float(score_refine)

                # refine 실패 시 wide 좌표 바로 채택하지 말고 hold
                return x, y, w, h, float(base_angle), float(score_wide)

            # ---------- 2차 실패 후 full-frame reacquire ----------
            pos_global, score_global = self._match_window(
                frame_gray8,
                0, 0, W - tw, H - th,
                margin=(0, 0),
                scale=0.35,
            )

            if pos_global is not None and score_global is not None and score_global >= 0.45:
                gx, gy, _, _ = pos_global

                pos_refine2, score_refine2 = self._match_window(
                    frame_gray8,
                    gx, gy, w, h,
                    margin=(self.search_margin_x, self.search_margin_y),
                    scale=1.0,
                )

                if pos_refine2 is not None and score_refine2 is not None and score_refine2 >= 0.55:
                    nx3, ny3, _, _ = pos_refine2
                    print(f"[DBG TRK] global_reacquire score={score_global:.3f} refine={score_refine2:.3f} pos=({nx3},{ny3})")
                    return nx3, ny3, w, h, float(base_angle), float(score_refine2)

            print(
                f"[DBG TRK] local={float(maxv):.3f} "
                f"wide={float(score_wide) if score_wide is not None else -1.0:.3f} "
                f"global={float(score_global) if score_global is not None else -1.0:.3f} "
                f"return_hold=({x},{y})"
            )
            best_low = max(
                float(maxv) if maxv is not None else 0.0,
                float(score_wide) if score_wide is not None else 0.0,
                float(score_global) if score_global is not None else 0.0,
            )
            return x, y, w, h, float(base_angle), best_low

        # 회전은 누적 angle 기준이 아니라 base_angle 기준 절대각 탐색
        best = (x, y, w, h, float(base_angle))
        best_score = -1.0

        a0 = max(-float(max_abs_angle), float(base_angle) - float(angle_range))
        a1 = min(+float(max_abs_angle), float(base_angle) + float(angle_range))
        angles = np.arange(a0, a1 + 0.001, angle_step, dtype=np.float32)

        for abs_angle in angles:
            tmpl = self._rotate_keep_size(self.template, float(abs_angle - float(base_angle)))
            res = cv2.matchTemplate(search, tmpl, self.method)
            _, maxv, _, maxloc = cv2.minMaxLoc(res)

            if maxv < self.thr:
                def _grad(img):
                    gx = cv2.Sobel(img, cv2.CV_32F, 1, 0, ksize=3)
                    gy = cv2.Sobel(img, cv2.CV_32F, 0, 1, ksize=3)
                    mag = cv2.magnitude(gx, gy)
                    mag = cv2.normalize(mag, None, 0, 255, cv2.NORM_MINMAX)
                    return mag.astype("uint8")

                search_g = _grad(search)
                tmpl_g = _grad(tmpl)

                res_g = cv2.matchTemplate(search_g, tmpl_g, cv2.TM_CCOEFF_NORMED)
                _, score_g, _, loc_g = cv2.minMaxLoc(res_g)

                if score_g > maxv:
                    maxv = score_g
                    maxloc = loc_g

            if maxv > best_score:
                nx = sx + int(maxloc[0])
                ny = sy + int(maxloc[1])
                best = (nx, ny, w, h, float(abs_angle))
                best_score = float(maxv)

        # ---------- 1차 실패 후 wide reacquire ----------
        pos_wide, score_wide = self._match_window(
            frame_gray8,
            x, y, w, h,
            margin=(self.wide_reacquire_margin_x, self.wide_reacquire_margin_y),
            scale=self.wide_reacquire_scale,
        )

        if pos_wide is not None and score_wide is not None and score_wide >= self.wide_reacquire_thr:
            rx, ry, _, _ = pos_wide

            # refine (정밀 재확인)
            pos_refine, score_refine = self._match_window(
                frame_gray8,
                rx, ry, w, h,
                margin=(self.search_margin_x, self.search_margin_y),
                scale=1.0,
            )

            # 회전 탐색 결과가 충분하면 바로 채택
            bx, by, _, _, ba = best
            if best_score >= self.thr:
                return bx, by, w, h, float(ba), float(best_score)

            # best 기준으로만 wide reacquire 1회
            pos_wide, score_wide = self._match_window(
                frame_gray8,
                bx, by, w, h,
                margin=(self.wide_reacquire_margin_x, self.wide_reacquire_margin_y),
                scale=self.wide_reacquire_scale,
            )

            if pos_wide is not None and score_wide is not None and score_wide >= self.wide_reacquire_thr:
                rx, ry, _, _ = pos_wide
                pos_refine, score_refine = self._match_window(
                    frame_gray8,
                    rx, ry, w, h,
                    margin=(self.search_margin_x, self.search_margin_y),
                    scale=1.0,
                )

                if pos_refine is not None and score_refine is not None and score_refine >= self.thr:
                    nx, ny, _, _ = pos_refine
                    print(f"[TRK] wide_reacquire score={score_wide:.3f} refine={score_refine:.3f} angle={ba:.2f}")
                    return nx, ny, w, h, float(ba), float(score_refine)

                print(f"[TRK] wide_reacquire (no refine) score={score_wide:.3f} angle={ba:.2f}")
                return bx, by, w, h, float(ba), float(score_wide)

            print(
                "[DBG TRK] "
                f"local={float(best_score):.3f} "
                f"wide={float(score_wide) if score_wide is not None else -1.0:.3f} "
                f"out=({int(bx)}, {int(by)}) angle={float(ba):.2f}"
            )
            return bx, by, w, h, float(ba), float(best_score if best_score > 0 else 0.0)