# src/capture/camera_gst.py
import cv2
import numpy as np

class CameraGST:
    """
    Minimal CameraGST with optional denoising and hot-pixel correction.

    Usage:
      cam = CameraGST(gst_pipeline,
                      auto_brightness=True,
                      target_mean=115,
                      denoise_method='median',   # 'none'|'median'|'nlmeans'|'hotpixel'
                      median_ksize=3,
                      nlm_h=10,
                      hp_collect_frames=50,
                      hp_thresh=30)
      cam.open(); frame = cam.read()
    """

    DEFAULT_PATTERNS = [
        cv2.COLOR_BayerBG2BGR,
        cv2.COLOR_BayerGB2BGR,
        cv2.COLOR_BayerRG2BGR,
        cv2.COLOR_BayerGR2BGR,
    ]

    def __init__(
        self,
        gst_pipeline: str,
        auto_brightness: bool = False,
        target_mean: float = 105.0,
        demosaic_cache: bool = True,
        patterns=None,
        # denoise/hotpixel options
        denoise_method: str = 'hotpixel',   # 'none','median','nlmeans','hotpixel'
        median_ksize: int = 53,
        nlm_h: float = 8.0,
        hp_collect_frames: int = 50,
        hp_thresh: float = 30.0,
    ):
        self.gst = gst_pipeline
        self.cap = None

        # Post-processing params
        self.auto_brightness = bool(auto_brightness)
        self.target_mean = float(target_mean)

        # demosaic cache (one-shot)
        self.demosaic_cache = bool(demosaic_cache)
        self._demosaic_code = None
        self._pattern_candidates = patterns or self.DEFAULT_PATTERNS

        # denoise/hotpixel
        self.denoise_method = denoise_method
        self.median_ksize = int(median_ksize) if median_ksize % 2 == 1 else int(median_ksize) + 1
        self.nlm_h = float(nlm_h)
        self.hp_collect_frames = int(hp_collect_frames)
        self.hp_thresh = float(hp_thresh)

        # hotpixel accumulator state
        self._hp_accum = None
        self._hp_count = 0
        self._hp_map = None  # boolean mask where True -> hot pixel

    def open(self):
        if self.cap is not None:
            return
        self.cap = cv2.VideoCapture(self.gst, cv2.CAP_GSTREAMER)
        if not self.cap.isOpened():
            self.cap = None
            raise RuntimeError("Camera open failed (GStreamer pipeline)")

    # ---------- helper: update hotpixel accumulator ----------
    def _accumulate_hotpixel(self, gray):
        """
        Accumulate gray frames to detect persistent hot pixels.
        After hp_collect_frames frames, compute mask where pixel is
        significantly brighter than local median.
        """
        if self._hp_accum is None:
            self._hp_accum = np.zeros_like(gray, dtype=np.float32)
            self._hp_count = 0
        self._hp_accum += gray.astype(np.float32)
        self._hp_count += 1

        if self._hp_count >= self.hp_collect_frames and self._hp_map is None:
            avg = (self._hp_accum / float(self._hp_count)).astype(np.uint8)
            # local median
            med = cv2.medianBlur(avg, 3)
            diff = cv2.subtract(avg, med).astype(np.float32)
            # threshold relative (hp_thresh) or relative to global mean
            gmean = float(avg.mean())
            # mask pixels that are consistently above local median by hp_thresh
            mask = (diff > self.hp_thresh).astype(np.uint8)
            # also require absolute brightness > global mean to avoid dark noise
            mask = np.logical_and(mask, avg > (gmean * 0.6))
            self._hp_map = mask.astype(np.uint8)
            # free accumulators
            self._hp_accum = None
            self._hp_count = 0
            # debug
            # print("[DBG CAMERA] hotpixel map built, count:", np.count_nonzero(self._hp_map))

    # ---------- helper: apply hotpixel correction ----------
    def _apply_hotpixel_map(self, frame):
        if self._hp_map is None:
            return frame
        # work on grayscale median and replace pixels in color image
        try:
            if len(frame.shape) == 3 and frame.shape[2] == 3:
                # compute median filtered image
                med = cv2.medianBlur(frame, 3)
                mask = self._hp_map.astype(bool)
                # replace masked pixels with median neighbors
                frame[mask] = med[mask]
            else:
                # grayscale fallback
                gray = frame if len(frame.shape) == 2 else cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                med = cv2.medianBlur(gray, 3)
                mask = self._hp_map.astype(bool)
                gray[mask] = med[mask]
                frame = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print("[DBG CAMERA] hp correction failed:", e)
        return frame

    def read(self):
        if self.cap is None:
            return None
        ret, frame = self.cap.read()
        if not ret or frame is None:
            return None

        # --- demosaic (one-shot cache) ---
        try:
            need_demosaic = (len(frame.shape) == 2) or (len(frame.shape) == 3 and frame.shape[2] == 1)
            if need_demosaic:
                if self.demosaic_cache and self._demosaic_code is not None:
                    try:
                        frame = cv2.cvtColor(frame, self._demosaic_code)
                    except Exception:
                        pass
                if self._demosaic_code is None:
                    best_code = None
                    best_score = -1.0
                    best_bgr = None
                    for code in self._pattern_candidates:
                        try:
                            bgr_try = cv2.cvtColor(frame, code)
                            chans = cv2.split(bgr_try)
                            std_sum = sum([float(c.std()) for c in chans])
                            hsv = cv2.cvtColor(bgr_try, cv2.COLOR_BGR2HSV)
                            sat_mean = float(hsv[:,:,1].mean())
                            score_try = std_sum + 0.5 * sat_mean
                            if score_try > best_score:
                                best_score = score_try
                                best_bgr = bgr_try
                                best_code = code
                        except Exception:
                            continue
                    if best_bgr is not None:
                        frame = best_bgr
                        if self.demosaic_cache:
                            self._demosaic_code = best_code
                    else:
                        if len(frame.shape) == 2:
                            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
                        else:
                            frame = cv2.cvtColor(frame[:,:,0], cv2.COLOR_GRAY2BGR)
        except Exception as e:
            print("[DBG CAMERA] demosaic error:", e)

        # --- build hotpixel map (if enabled) ---
        try:
            if self.denoise_method == 'hotpixel':
                # use grayscale average for hp detection
                gray_for_hp = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                if self._hp_map is None:
                    # accumulate
                    self._accumulate_hotpixel(gray_for_hp)
                else:
                    # apply map now
                    frame = self._apply_hotpixel_map(frame)
        except Exception as e:
            print("[DBG CAMERA] hotpixel processing error:", e)

        # --- simple auto-brightness (gain only) ---
        try:
            if self.auto_brightness:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if len(frame.shape) == 3 else frame
                mean_val = float(gray.mean()) if gray is not None else 0.0
                if mean_val > 0:
                    alpha = (self.target_mean / (mean_val + 1e-6))
                    alpha = max(1.0, min(alpha, 2.0))   # do NOT darken: clamp min to 1.0
                    if abs(alpha - 1.0) > 0.01:
                        frame = cv2.convertScaleAbs(frame, alpha=alpha, beta=0)
        except Exception as e:
            print("[DBG CAMERA] simple postproc error:", e)

        # --- optional denoise (spatial) ---
        try:
            if self.denoise_method == 'median':
                # small median removes isolated bright dots; cheap and effective
                if self.median_ksize > 1:
                    frame = cv2.medianBlur(frame, self.median_ksize)
            elif self.denoise_method == 'nlmeans':
                # non-local means denoising - slower but preserves edges
                # h controls strength for luminance; hColor for color
                frame = cv2.fastNlMeansDenoisingColored(frame, None, h=self.nlm_h, hColor=self.nlm_h, templateWindowSize=7, searchWindowSize=21)
            # 'hotpixel' handled earlier; 'none' -> no denoise
        except Exception as e:
            print("[DBG CAMERA] denoise error:", e)

        return frame

    def release(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None