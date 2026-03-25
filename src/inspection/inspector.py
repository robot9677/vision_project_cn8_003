import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .recipe import load_recipe, get_roi_cfg, get_inspection_cfgs, has_explicit_inspections, has_inspection_for_roi, save_recipe
from .analyzers import run_analyzer
from .preprocess import normalize_by_roi
from .temporal import TemporalMeanFilter
from .roi_tracker import ROITracker
from .aligner import MultiAnchorAligner
# add near top of file
from inspection.score import combined_score
from inspection.toolchain import run_toolchain
from inspection.tools_enhance import register_enhance_tools
from inspection.tools_measure import register_measure_tools
from inspection.tools_locate import register_locate_tools
from inspection.tools_identify import register_identify_tools
from inspection.tools_measure_washer import run_washer_presence

def _run_presence_job(crop, cfg):
    params = cfg if isinstance(cfg, dict) else {}

    blur_ksize = int(params.get("blur_ksize", 3))
    if blur_ksize < 1:
        blur_ksize = 1
    if blur_ksize % 2 == 0:
        blur_ksize += 1

    threshold_mode = str(params.get("threshold_mode", "fixed")).strip().lower()
    threshold = float(params.get("threshold", 128))
    offset = float(params.get("offset", 0.0))
    invert = bool(params.get("invert", False))

    morph_open = int(params.get("morph_open", 0))
    morph_close = int(params.get("morph_close", 0))
    min_area = int(params.get("min_area", 0))

    img = crop
    if blur_ksize > 1:
        img = cv2.GaussianBlur(img, (blur_ksize, blur_ksize), 0)

    mean_val = float(np.mean(img))

    if threshold_mode == "mean_offset":
        th_value = mean_val + offset
    elif threshold_mode == "otsu":
        th_value = 0.0
    else:
        th_value = threshold

    th_flag = cv2.THRESH_BINARY_INV if invert else cv2.THRESH_BINARY

    if threshold_mode == "otsu":
        _ret, bw = cv2.threshold(img, 0, 255, th_flag | cv2.THRESH_OTSU)
        th_value = float(_ret)
    else:
        _ret, bw = cv2.threshold(img, float(th_value), 255, th_flag)

    if morph_open > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_open, morph_open))
        bw = cv2.morphologyEx(bw, cv2.MORPH_OPEN, k)

    if morph_close > 0:
        k = cv2.getStructuringElement(cv2.MORPH_RECT, (morph_close, morph_close))
        bw = cv2.morphologyEx(bw, cv2.MORPH_CLOSE, k)

    num_labels, labels, stats, _centroids = cv2.connectedComponentsWithStats(bw, connectivity=8)

    kept_area = 0
    kept_count = 0
    out = np.zeros_like(bw)

    for i in range(1, num_labels):
        area = int(stats[i, cv2.CC_STAT_AREA])
        if area < min_area:
            continue
        out[labels == i] = 255
        kept_area += area
        kept_count += 1

    area_ratio = kept_area / float(out.shape[0] * out.shape[1]) if out.size > 0 else 0.0

    metrics = {
        "presence_area": int(kept_area),
        "presence_count": int(kept_count),
        "presence_ratio": float(area_ratio),
        "th_value": float(th_value),
        "_last_image": out,
    }

    ok = True
    reason = "OK"
    return ok, metrics, reason

def _run_qr_job(crop, cfg):
    detector = cv2.QRCodeDetector()

    def _try_decode(img):
        data, points, _straight = detector.detectAndDecode(img)
        detected = bool(data)
        if not detected and points is not None:
            detected = True
        return detected, str(data or ""), points

    variants = []

    base = crop
    if base is None or base.size == 0:
        return False, {"qr_detected": False, "qr_text": "", "_last_image": crop}, "QR_EMPTY"

    # 1) 원본
    variants.append(("raw", base))

    # 2) 2배 확대
    up2 = cv2.resize(base, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("up2", up2))

    # 3) 3배 확대
    up3 = cv2.resize(base, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("up3", up3))

    # 4) 히스토그램 평활화 + 확대
    eq = cv2.equalizeHist(base)
    eq2 = cv2.resize(eq, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    variants.append(("eq2", eq2))

    # 5) Gaussian blur 후 OTSU
    blur = cv2.GaussianBlur(base, (3, 3), 0)
    _, otsu = cv2.threshold(blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    otsu2 = cv2.resize(otsu, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    variants.append(("otsu2", otsu2))

    # 6) 반전 OTSU + 확대
    otsu_inv = cv2.bitwise_not(otsu)
    otsu_inv2 = cv2.resize(otsu_inv, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_NEAREST)
    variants.append(("otsu_inv2", otsu_inv2))

    best_img = base
    best_name = "raw"
    qr_detected = False
    qr_text = ""

    for name, img in variants:
        detected, text, points = _try_decode(img)
        if detected:
            qr_detected = True
            qr_text = text
            best_img = img
            best_name = name
            break

    metrics = {
        "qr_detected": bool(qr_detected),
        "qr_text": str(qr_text or ""),
        "qr_variant": best_name,
        "_last_image": best_img,
    }

    ok = True
    reason = "OK"
    return ok, metrics, reason

@dataclass
class ROIResult:
    roi_id: Any
    ok: bool
    reason: str
    metrics: Dict[str, Any]

def _job_eval_toolchain(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    job_ok = bool(ok)
    job_reason = "OK" if job_ok else (reason or "FAIL")
    return job_ok, job_reason


def _job_eval_mean_threshold(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    default_min = float(recipe_default.get("min_mean", 0.0))
    default_max = float(recipe_default.get("max_mean", 255.0))

    min_mean = float(cfg.get("min_mean", default_min))
    max_mean = float(cfg.get("max_mean", default_max))

    use_avg5 = bool(runtime_cfg.get("auto_inspect_avg5", False))
    mean_val = float(metrics.get("mean", 0.0)) if use_avg5 else float(metrics.get("mean_raw", 0.0))

    job_ok = (min_mean <= mean_val <= max_mean)
    job_reason = "OK" if job_ok else ("LOW_MEAN" if mean_val < min_mean else "HIGH_MEAN")
    return bool(job_ok), job_reason


def _job_eval_score_threshold(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    default_score_thresh = float(recipe_default.get("score_threshold", 0.25))
    score_thresh = float(cfg.get("score_threshold", default_score_thresh))
    score_val = float(metrics.get("score", 0.0))

    job_ok = score_val >= score_thresh
    job_reason = "OK" if job_ok else "LOW_SCORE"
    return bool(job_ok), job_reason

def _job_eval_presence(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    min_ratio = float(cfg.get("min_ratio", 0.0))
    max_ratio = float(cfg.get("max_ratio", 1.0))
    min_count = int(cfg.get("min_count", 0))
    max_count = int(cfg.get("max_count", 999999))

    ratio = float(metrics.get("presence_ratio", 0.0))
    count = int(metrics.get("presence_count", 0))

    ratio_ok = (min_ratio <= ratio <= max_ratio)
    count_ok = (min_count <= count <= max_count)

    job_ok = ratio_ok and count_ok

    if not ratio_ok:
        if ratio < min_ratio:
            return False, "PRESENCE_RATIO_LOW"
        return False, "PRESENCE_RATIO_HIGH"

    if not count_ok:
        if count < min_count:
            return False, "PRESENCE_COUNT_LOW"
        return False, "PRESENCE_COUNT_HIGH"

    return bool(job_ok), "OK"

def _job_eval_qr_presence(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    detected = bool(metrics.get("qr_detected", False))
    if detected:
        return True, "OK"
    return False, "QR_NOT_FOUND"

def _job_eval_washer(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    edge_count = int(metrics.get("edge_count", 0))
    mean_raw = float(metrics.get("mean_raw", 0.0))
    peak_count = int(metrics.get("peak_count", 0))

    min_edge = int(cfg.get("min_edge", 170))
    min_mean = float(cfg.get("min_mean", 38.0))
    min_peak = int(cfg.get("min_peak", 2))

    # 1차: 존재
    if edge_count < min_edge or mean_raw < min_mean:
        return False, "WASHER_MISSING"

    # 2차: 개수
    if peak_count < min_peak:
        return False, "WASHER_COUNT_LOW"

    return True, "OK"

JOB_EVALUATORS = {
    "toolchain": _job_eval_toolchain,
    "mean_threshold": _job_eval_mean_threshold,
    "score_threshold": _job_eval_score_threshold,
    "presence": _job_eval_presence,
    "qr_presence": _job_eval_qr_presence,
    "washer_presence": _job_eval_washer, 
}

def _run_toolchain_job(crop, cfg):
    return run_toolchain(crop, cfg)


def _run_analyzer_job(crop, cfg):
    return run_analyzer(crop, cfg)

JOB_RUNNERS = {
    "toolchain": _run_toolchain_job,
    "mean_threshold": _run_analyzer_job,
    "score_threshold": _run_analyzer_job,
    "presence": _run_presence_job,
    "qr_presence": _run_qr_job,
    "washer_presence": run_washer_presence,
}

class Inspector:
    def __init__(self, roi_mgr, recipe_path: str, logs_root: str, runtime_cfg=None):
        self.roi_mgr = roi_mgr
        self.recipe_path = recipe_path
        self.logs_root = logs_root
        self.recipe = load_recipe(recipe_path)
        print("[RECIPE]", "STATIC", recipe_path)
        self.mean_filters = {}
        self.runtime_cfg = runtime_cfg or {}
        self.debug_view_enabled = bool(self.runtime_cfg.get("debug_view_enabled", True))
        self.debug_view_roi_id = str(self.runtime_cfg.get("debug_view_roi_id", "1"))
        self.debug_view_scale = float(self.runtime_cfg.get("debug_view_scale", 1))
        self.tracker = ROITracker(
            search_margin=int(self.runtime_cfg.get("tracker_search_margin", 80)),
            thr=float(self.runtime_cfg.get("tracker_thr", 0.70)),
            reacquire_margin=int(self.runtime_cfg.get("tracker_reacquire_margin", 220)),
            reacquire_scale=float(self.runtime_cfg.get("tracker_reacquire_scale", 0.5)),
        )
        self.aligner = MultiAnchorAligner(
            runtime_cfg=self.runtime_cfg,
            product_profile=(self.runtime_cfg.get("_product_profile") or {}),
            project_root=os.path.abspath(os.path.join(os.path.dirname(recipe_path), "..", "..")),
        )
        self.debug_images = {}
        self.debug_tiles = {}
        self.baseline_path = os.path.join(os.path.dirname(recipe_path), "baseline_profile.json")
        if os.path.exists(self.baseline_path):
            with open(self.baseline_path, "r") as f:
                self.baseline = json.load(f)
        else:
            self.baseline = None

        register_enhance_tools()
        register_measure_tools()
        register_locate_tools()
        register_identify_tools()

    def _run_inspection_job(
        self,
        crop,
        cfg,
        recipe_default,
        runtime_cfg,
        mean_filter,
        norm_gain,
        roi_dx,
        roi_dy,
        roi_dangle,
        pose,
        trk_score,
    ):
        job_type = (cfg.get("type") or "").strip().lower()
        # --- washer 전용 tracking 제한 ---
        orig_margin = None

        if job_type == "washer_presence":
            orig_margin = getattr(self.tracker, "search_margin", None)
            self.tracker.search_margin = int(cfg.get("tracker_margin", 50))

        runner = JOB_RUNNERS.get(job_type, _run_analyzer_job)
        ok, metrics, reason = runner(crop, cfg)

        if metrics is None:
            metrics = {}

        mean_raw = float(np.mean(crop))
        metrics["mean_raw"] = mean_raw
        metrics["mean"] = mean_filter.update(mean_raw)

        need_score = str(cfg.get("type", "")).lower() in ("mean_score", "score_threshold", "texture_score")
        if need_score:
            try:
                score = combined_score(crop)
            except Exception:
                score = 0.0
            metrics["score"] = float(score)

        job_type = (cfg.get("type") or "").strip().lower()
        evaluator = JOB_EVALUATORS.get(job_type, _job_eval_toolchain)
        job_ok, job_reason = evaluator(
            ok=ok,
            metrics=metrics,
            reason=reason,
            cfg=cfg,
            recipe_default=recipe_default,
            runtime_cfg=runtime_cfg,
        )

        metrics["norm_gain"] = float(norm_gain)
        metrics["dx"] = roi_dx
        metrics["dy"] = roi_dy
        metrics["dangle"] = float(roi_dangle)
        metrics["trk_score"] = float(pose.get("score", trk_score))
        metrics["align_anchor_id"] = pose.get("anchor_id")
        metrics["inspection_id"] = cfg.get("id", "job")

        # --- tracker 복구 ---
        if orig_margin is not None:
            self.tracker.search_margin = orig_margin

        return job_ok, metrics, job_reason, job_type

    def _get_mean_filter(self, roi_id):
        key = str(roi_id)
        if key not in self.mean_filters:
            self.mean_filters[key] = TemporalMeanFilter(win=5)
        return self.mean_filters[key]

    def reload_recipe(self):
        self.recipe = load_recipe(self.recipe_path)

    def _show_debug_view(self, roi_id, raw_crop=None, last_img=None):
        if not self.debug_view_enabled:
            return

        def _to_bgr(im):
            if im is None or not isinstance(im, np.ndarray) or im.size == 0:
                return None
            out = im.copy()
            if out.ndim == 2:
                out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
            return out

        raw_vis = _to_bgr(raw_crop)
        last_vis = _to_bgr(last_img)

        cell_w = 110
        cell_h = 70

        def _fit_cell(im, title, color):
            canvas = np.zeros((cell_h, cell_w, 3), dtype=np.uint8)
            if im is not None:
                h, w = im.shape[:2]
                scale = min((cell_w - 8) / max(1, w), (cell_h - 28) / max(1, h))
                nw = max(1, int(w * scale))
                nh = max(1, int(h * scale))
                resized = cv2.resize(im, (nw, nh), interpolation=cv2.INTER_NEAREST)
                x0 = (cell_w - nw) // 2
                y0 = 24 + (cell_h - 24 - nh) // 2
                canvas[y0:y0+nh, x0:x0+nw] = resized
            cv2.putText(canvas, title, (4, 10), cv2.FONT_HERSHEY_SIMPLEX, 0.2, color, 1)
            cv2.rectangle(canvas, (0, 0), (cell_w - 1, cell_h - 1), (60, 60, 60), 1)
            return canvas

        left = _fit_cell(raw_vis, f"ROI{roi_id} RAW", (0, 255, 255))
        right = _fit_cell(last_vis, f"ROI{roi_id} LAST", (0, 255, 0))
        pair = cv2.hconcat([left, right])

        self.debug_tiles[str(roi_id)] = pair

        keys = sorted(self.debug_tiles.keys(), key=lambda x: int(x))
        pairs = [self.debug_tiles[k] for k in keys]

        per_row = 2
        blank = np.zeros_like(pair)
        rows = []

        for i in range(0, len(pairs), per_row):
            row = pairs[i:i+per_row]
            while len(row) < per_row:
                row.append(blank.copy())
            rows.append(cv2.hconcat(row))

        grid = cv2.vconcat(rows)
        cv2.namedWindow("ROI DEBUG", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("ROI DEBUG", 600, 350)
        cv2.imshow("ROI DEBUG", grid)

    def _crop_rotated(self, frame_gray8, roi, dx=0, dy=0, dangle=0.0):
        H, W = frame_gray8.shape[:2]

        x = float(roi.get("x", 0)) + float(dx)
        y = float(roi.get("y", 0)) + float(dy)
        w = max(1, int(roi.get("w", 1)))
        h = max(1, int(roi.get("h", 1)))
        angle = float(roi.get("angle", 0.0)) + float(dangle)

        cx = x + w / 2.0
        cy = y + h / 2.0

        rect = ((cx, cy), (w, h), angle)
        box = cv2.boxPoints(rect).astype(np.float32)

        min_x = max(0, int(np.floor(np.min(box[:, 0]))))
        min_y = max(0, int(np.floor(np.min(box[:, 1]))))
        max_x = min(W, int(np.ceil(np.max(box[:, 0]))))
        max_y = min(H, int(np.ceil(np.max(box[:, 1]))))

        if max_x <= min_x or max_y <= min_y:
            return None

        patch = frame_gray8[min_y:max_y, min_x:max_x]
        if patch is None or patch.size == 0:
            return None

        local_cx = cx - min_x
        local_cy = cy - min_y

        M = cv2.getRotationMatrix2D((local_cx, local_cy), angle, 1.0)
        rotated = cv2.warpAffine(
            patch,
            M,
            (patch.shape[1], patch.shape[0]),
            flags=cv2.INTER_LINEAR,
            borderMode=cv2.BORDER_REPLICATE,
        )

        crop = cv2.getRectSubPix(rotated, (w, h), (local_cx, local_cy))
        return crop

    def inspect(self, frame_gray8: np.ndarray, auto_mode=False):
        if self.debug_view_enabled:
            self.debug_tiles = {}
            try:
                cv2.destroyWindow("ROI DEBUG")
            except:
                pass

        results: Dict[str, ROIResult] = {}

        ref = self.roi_mgr.get(1)
        norm_gain = 1.0
        dx = dy = 0
        dangle = 0.0
        trk_score = 0.0
        align_result = None

        # 1) ref 기반 정규화 후 align 단계 수행
        if ref is not None:
            ref_id = ref["id"]
            ref_crop_raw = self.roi_mgr.crop(frame_gray8, ref_id)

            use_normalize = bool(self.runtime_cfg.get("normalize_enabled", False))
            if use_normalize and ref_crop_raw is not None and ref_crop_raw.size > 0:
                target_mean = float(self.runtime_cfg.get("normalize_target_mean", 50.0))
                frame_gray8, norm_gain = normalize_by_roi(frame_gray8, ref_crop_raw, target_mean=target_mean)
            else:
                norm_gain = 1.0

        use_tracker = bool(self.runtime_cfg.get("enable_tracker", True))

        # washer 검사일 때 tracker 끄기
        if any(cfg.get("type") == "washer_presence" for cfg in get_inspection_cfgs(self.recipe, 3)):
            use_tracker = False

        if use_tracker and getattr(self, "aligner", None) is not None:
            align_result = self.aligner.estimate(frame_gray8, self.roi_mgr)
            g = align_result.get("global") or {}
            dx = int(g.get("dx", 0))
            dy = int(g.get("dy", 0))
            dangle = float(g.get("dangle", 0.0))
            trk_score = float(g.get("score", 0.0))
            if not auto_mode:
                anchors = align_result.get("anchors") or []
                if anchors:
                    dbg = " ".join(
                        f"{a.get('id')}[ok={a.get('ok')} dx={a.get('dx')} dy={a.get('dy')} da={a.get('dangle'):.2f} sc={a.get('score'):.3f}]"
                        for a in anchors
                    )
                    print(f"[DBG ALIGN] {dbg}")
        else:
            align_result = {"per_roi": {}, "global": {"dx": 0, "dy": 0, "dangle": 0.0, "score": 0.0}}

        # 2) 모든 ROI는 Δ만 적용해서 crop (안정)
        H, W = frame_gray8.shape[:2]

        anchor_roi_ids = {int(a.get("roi_id", 0)) for a in getattr(self.aligner, "_anchors", [])}

        profile_rois = (self.runtime_cfg.get("_product_profile", {}) or {}).get("rois") or []
        roi_types = {
            int(r.get("id")): str(r.get("type", "")).strip().lower()
            for r in profile_rois
            if r.get("id") is not None
        }

        use_explicit_inspections = has_explicit_inspections(self.recipe)

        for roi in getattr(self.roi_mgr, "rois", []):
            roi_id = int(roi.get("id"))
            key = str(roi_id)

            roi_type = roi_types.get(roi_id, "")
            roi_has_job = has_inspection_for_roi(self.recipe, roi_id)

            if use_explicit_inspections:
                if not roi_has_job:
                    continue
            else:
                if roi_type != "inspect":
                    continue

            pose = (align_result or {}).get("per_roi", {}).get(int(roi_id), {})
            roi_dx = int(pose.get("dx", 0))
            roi_dy = int(pose.get("dy", 0))
            roi_dangle = float(pose.get("dangle", 0.0))

            crop = self._crop_rotated(frame_gray8, roi, dx=roi_dx, dy=roi_dy, dangle=roi_dangle)
            
            if crop is None or crop.size == 0:
                results[key] = ROIResult(roi_id=roi_id, ok=False, reason="EMPTY_CROP", metrics={})
                continue

            if crop is None or crop.size == 0:
                if not auto_mode:
                    print(f"[DBG INSPECT] ROI{roi_id} EMPTY_CROP")
            # cfg = get_roi_cfg(self.recipe, roi_id)
            # ok, metrics, reason = run_analyzer(crop, cfg)

            inspection_cfgs = get_inspection_cfgs(self.recipe, roi_id)
            cfg = inspection_cfgs[0] if inspection_cfgs else get_roi_cfg(self.recipe, roi_id)
            # === 분석 및 mean+score 기반 판정 통합 ===
            job_results = []
            merged_metrics = {}
            final_ok = True
            final_reason = "OK"
            last_metrics = {}

            for cfg in inspection_cfgs:
                job_ok, metrics, job_reason, roi_type = self._run_inspection_job(
                    crop=crop,
                    cfg=cfg,
                    recipe_default=self.recipe.get("default", {}),
                    runtime_cfg=self.runtime_cfg,
                    mean_filter=self._get_mean_filter(roi_id),
                    norm_gain=norm_gain,
                    roi_dx=roi_dx,
                    roi_dy=roi_dy,
                    roi_dangle=roi_dangle,
                    pose=pose,
                    trk_score=trk_score,
                )

                if "tools" in cfg and cfg.get("tools"):
                    if not auto_mode and roi_id == 1:
                        dbg_dir = os.path.join(self.logs_root, "_dbg")
                        os.makedirs(dbg_dir, exist_ok=True)
                        tool_img = metrics.get("_last_image")
                        if isinstance(tool_img, np.ndarray) and tool_img.size > 0:
                            cv2.imwrite(
                                os.path.join(dbg_dir, f"roi1_{int(time.time()*1000)}_{'OK' if job_ok else 'NG'}.png"),
                                tool_img,
                            )

                job_results.append(
                    {
                        "id": cfg.get("id", f"ROI{roi_id}"),
                        "type": roi_type,
                        "ok": bool(job_ok),
                        "reason": job_reason,
                        "metrics": {k: v for k, v in metrics.items() if not k.startswith("_")},
                    }
                )

                if not job_ok and final_ok:
                    final_ok = False
                    final_reason = job_reason

                for mk, mv in metrics.items():
                    if mk.startswith("_"):
                        continue
                    if mk not in merged_metrics:
                        merged_metrics[mk] = mv

                last_metrics = metrics

            if not final_ok:
                reason = final_reason
            else:
                reason = "OK"

            metrics = dict(merged_metrics)
            metrics["_inspections"] = job_results
            if isinstance(last_metrics, dict) and last_metrics.get("_last_image") is not None:
                metrics["_last_image"] = last_metrics.get("_last_image")

            self._show_debug_view(
                roi_id=roi_id,
                raw_crop=crop,
                last_img=metrics.get("_last_image"),
            )

            if roi_id not in anchor_roi_ids:
                b_ok, b_reason = self._check_baseline(roi_id, metrics)
                if not b_ok:
                    final_ok = False
                    reason = b_reason

            results[key] = ROIResult(roi_id=roi_id, ok=final_ok, reason=reason, metrics=metrics)

            if not auto_mode:
                print(
                    f"[DBG ROI{roi_id}] ok={final_ok} reason={reason} "
                    f"blob={metrics.get('blob_count')} "
                    f"areas={metrics.get('blob_areas_kept')} "
                    f"boxes={metrics.get('blob_boxes_kept')} "
                    f"zone={metrics.get('count_zone')} "
                    f"th={metrics.get('th_value')} "
                    f"white_ratio={metrics.get('white_ratio')} "
                    f"dark_ratio={metrics.get('dark_ratio')} "
                    f"qr_detected={metrics.get('qr_detected')} "
                    f"qr_text={metrics.get('qr_text')} "
                    f"edge_count={metrics.get('edge_count')} "
                    f"band_h={metrics.get('band_h')} "
                    f"mean_raw={metrics.get('mean_raw')} "
                    f"mean={metrics.get('mean')} "
                    f"norm_gain={metrics.get('norm_gain')} "
                    f"dx={metrics.get('dx')} dy={metrics.get('dy')} "
                    f"dangle={metrics.get('dangle')} "
                    f"trk_score={metrics.get('trk_score')}"
                )
                dbg_path = os.path.join(self.logs_root, f"roi{roi_id}_last.png")
                last_img = metrics.get("_last_image")
                if not auto_mode and last_img is not None:
                    cv2.imwrite(dbg_path, last_img)
                    print(f"[DBG SAVE] {dbg_path}")

                if metrics.get("_tool_steps") is not None:
                    print(f"[DBG TOOLS ROI{roi_id}] {metrics.get('_tool_steps')}")

        # --- overall decision by recipe ---
        decision = (self.recipe.get("decision") or {})
        mode = (decision.get("mode") or "any_fail_is_ng").strip().lower()

        oks = [r.ok for r in results.values()]
        if not oks:
            overall_ok = False
        else:
            if mode == "any_fail_is_ng":
                overall_ok = all(oks)
            elif mode == "majority_ok":
                overall_ok = (sum(1 for v in oks if v) >= (len(oks) / 2))
            elif mode == "allow_fail_count":
                max_fail = int(decision.get("max_fail", 0))
                fail_cnt = sum(1 for v in oks if not v)
                overall_ok = (fail_cnt <= max_fail)
            else:
                # fallback
                overall_ok = all(oks)

        if not auto_mode:
            print(f"[DBG] overall decision by recipe : {mode}")
        return overall_ok, results


    def save_run(self, frame_gray8: np.ndarray, overlay_bgr: np.ndarray, overall_ok: bool, results: Dict[str, ROIResult]) -> str:
        day = time.strftime("%Y%m%d")
        ts  = time.strftime("%H%M%S")
        mmm = int((time.time() * 1000) % 1000)
        run_dir = os.path.join(self.logs_root, day, f"{ts}_{mmm:03d}")
        os.makedirs(run_dir, exist_ok=True)

        cv2.imwrite(os.path.join(run_dir, "raw.png"), frame_gray8)
        cv2.imwrite(os.path.join(run_dir, "overlay.png"), overlay_bgr)

        out = {
            "overall_ok": bool(overall_ok),
            "ts": time.time(),
            "results": {
                k: {
                    "roi_id": str(v.roi_id),
                    "ok": bool(v.ok),
                    "reason": v.reason,
                    "metrics": {k: v2 for k, v2 in v.metrics.items() if not isinstance(v2, np.ndarray)},
                } for k, v in results.items()
            }
        }
        with open(os.path.join(run_dir, "result.json"), "w", encoding="utf-8") as f:
            json.dump(out, f, ensure_ascii=False, indent=2)

        return run_dir
    
    # def save_recipe(path: str, recipe: Dict[str, Any]) -> None:
    #     import os, json
    #     os.makedirs(os.path.dirname(path), exist_ok=True)
    #     with open(path, "w", encoding="utf-8") as f:
    #         json.dump(recipe, f, ensure_ascii=False, indent=2)

    def autotune_recipe_from_frame(self, frame_gray8, save_path=None):
        import copy
        target_mean = float(self.runtime_cfg.get("autotune_target_mean", 50.0))
        margin = float(self.runtime_cfg.get("autotune_margin", 10.0))
        save_path = save_path or self.recipe_path
        
        """
        현재 프레임 기준으로 ROI별 mean을 읽고
        recipe_static.json(overrides)에 ROI별 min/max를 자동 생성해서 저장
        """
        # 1) 기준 ROI로 정규화(현재 inspect랑 동일 로직)
        ref = self.roi_mgr.get_selected()
        if ref is not None:
            ref_crop = self.roi_mgr.crop(frame_gray8, ref["id"])
            frame_n, _gain = normalize_by_roi(frame_gray8, ref_crop, target_mean=target_mean)
        else:
            frame_n = frame_gray8

        overrides = {}
        for roi in getattr(self.roi_mgr, "rois", []):
            roi_id = roi.get("id")
            crop = self.roi_mgr.crop(frame_n, roi_id)
            if crop is None or crop.size == 0:
                continue
            m = float(np.mean(crop))
            mn = max(0.0, m - margin)
            mx = min(255.0, m + margin)
            cfg = get_roi_cfg(self.recipe, roi_id)

            if "tools" in cfg:
                roi_cfg = copy.deepcopy(cfg)

                for step in roi_cfg.get("tools", []):
                    tool_name = str(step.get("tool", "")).strip().lower()
                    params = step.get("params") or {}

                    if tool_name == "measure.blob_count":
                        ok_bt, metrics_bt, _reason_bt = run_toolchain(crop, {
                            "tools": roi_cfg.get("tools", []),
                            "tool_decision": roi_cfg.get("tool_decision", "all_ok"),
                        })

                        blob_count = int(metrics_bt.get("blob_count", 0))
                        areas = metrics_bt.get("blob_areas_kept") or []

                        # expected 는 자동 변경하지 않음
                        # 정상 기준 개수는 사용자가 직접 정하거나 기존 값을 유지

                        if areas:
                            params["area_min"] = int(max(1, min(areas) * 0.7))
                            params["area_max"] = int(max(areas) * 1.3)

                        step["params"] = params

                overrides[f"ROI{roi_id}"] = roi_cfg
            else:
                overrides[f"ROI{roi_id}"] = {
                    "type": "mean_threshold",
                    "min_mean": float(mn),
                    "max_mean": float(mx),
                }
        # 기존 recipe를 베이스로 복사해서, overrides만 교체
        base = self.recipe if isinstance(self.recipe, dict) else {}
        recipe = copy.deepcopy(base)

        # default는 없으면 넣고, 있으면 유지(원하면 여기서만 type 보정)
        recipe.setdefault("default", {"type": "mean_threshold", "min_mean": 0.0, "max_mean": 255.0})

        # 핵심: AUTO는 overrides만 갱신
        recipe["overrides"] = overrides

        # decision은 절대 건드리지 않음(없으면 기본값만 세팅)
        recipe.setdefault("decision", {"mode": "any_fail_is_ng"})

        # print("[DBG AUTO] decision(before save) =", recipe.get("decision"))
        save_recipe(save_path, recipe)
        # print("[DBG AUTO] saved to", save_path)
        # print("[DBG AUTO] decision(after save read) =", load_recipe(save_path).get("decision"))
        self.recipe = recipe  # 즉시 반영
        return recipe
    
    def reset_tracker_template(self):
        self.tracker.template = None
        if getattr(self, "aligner", None) is not None:
            self.aligner.reset_templates()

    def log_result(self, overall_ok, results):
        os.makedirs(self.logs_root, exist_ok=True)
        ts = time.strftime("%Y%m%d_%H%M%S")
        path = os.path.join(self.logs_root, f"inspect_{ts}.json")

        payload = {
            "ts": ts,
            "overall_ok": bool(overall_ok),
            "results": {
                str(k): {
                    "ok": bool(v.ok) if hasattr(v, "ok") else bool(v.get("ok")),
                    "reason": (v.reason if hasattr(v, "reason") else v.get("reason","")),
                    "metrics": (v.metrics if hasattr(v, "metrics") else v.get("metrics", {})),
                }
                for k, v in (results or {}).items()
            }
        }
        with open(path, "w") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
        
        self._prune_logs(max_keep=200)

    def _prune_logs(self, max_keep=200, max_mb=300):
        try:
            os.makedirs(self.logs_root, exist_ok=True)

            # 1) inspect_*.json만 정리(최신 max_keep 유지)
            jsons = []
            for fn in os.listdir(self.logs_root):
                if fn.startswith("inspect_") and fn.endswith(".json"):
                    p = os.path.join(self.logs_root, fn)
                    jsons.append(p)

            jsons.sort(key=lambda p: os.path.getmtime(p), reverse=True)

            for p in jsons[max_keep:]:
                try:
                    os.remove(p)
                except Exception:
                    pass

            # 2) 폴더 용량 제한 (오래된 것부터 삭제)
            max_bytes = int(max_mb * 1024 * 1024)
            files = []
            total = 0

            for root, _, fns in os.walk(self.logs_root):
                for fn in fns:
                    p = os.path.join(root, fn)
                    try:
                        st = os.stat(p)
                    except Exception:
                        continue
                    files.append((st.st_mtime, p, st.st_size))
                    total += st.st_size

            if total <= max_bytes:
                return

            files.sort(key=lambda t: t[0])  # 오래된 순
            for _mtime, p, sz in files:
                try:
                    os.remove(p)
                    total -= sz
                except Exception:
                    pass
                if total <= max_bytes:
                    break

        except Exception:
            pass

    def _check_baseline(self, roi_id, metrics):
        if not self.baseline:
            return True, None

        roi_name = f"ROI{roi_id}"
        if roi_name not in self.baseline:
            return True, None

        data = self.baseline[roi_name]

        # ROI2~5
        if "dark_ratio" in data:
            v = metrics.get("dark_ratio")
            if v is None:
                return True, None

            low = data["dark_ratio"]["low"]
            high = data["dark_ratio"]["high"]

            if v < low:
                return False, "BASELINE_LOW"
            if v > high:
                return False, "BASELINE_HIGH"

            return True, None

        # ROI6
        if "blob_count" in data:
            v = metrics.get("blob_count")
            if v is None:
                v = metrics.get("blob")

            if v is None:
                return True, None

            low = data["blob_count"]["low"]
            high = data["blob_count"]["high"]

            if v < low:
                return False, "BASELINE_LOW"
            if v > high:
                return False, "BASELINE_HIGH"

            return True, None

        return True, None