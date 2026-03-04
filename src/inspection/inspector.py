import os
import json
import time
from dataclasses import dataclass
from typing import Any, Dict, Tuple

import cv2
import numpy as np

from .recipe import load_recipe, get_roi_cfg, save_recipe
from .analyzers import run_analyzer
from .preprocess import normalize_by_roi
from .temporal import TemporalMeanFilter
from .roi_tracker import ROITracker
# add near top of file
from inspection.score import combined_score

@dataclass
class ROIResult:
    roi_id: Any
    ok: bool
    reason: str
    metrics: Dict[str, Any]

class Inspector:
    def __init__(self, roi_mgr, recipe_path: str, logs_root: str):
        self.roi_mgr = roi_mgr
        self.recipe_path = recipe_path
        self.logs_root = logs_root
        auto_path = os.path.join(os.path.dirname(recipe_path), "recipe_auto.json")
        self.recipe = load_recipe(auto_path if os.path.exists(auto_path) else recipe_path)
        print("[RECIPE]", "AUTO" if os.path.exists(auto_path) else "STATIC", (auto_path if os.path.exists(auto_path) else recipe_path))
        self.mean_filter = TemporalMeanFilter(win=5)
        self.tracker = ROITracker(search_margin=20, thr=0.6)


    def reload_recipe(self):
        self.recipe = load_recipe(self.recipe_path)

    def inspect(self, frame_gray8: np.ndarray) -> Tuple[bool, Dict[str, ROIResult]]:
        results: Dict[str, ROIResult] = {}

        ref = self.roi_mgr.get_selected()
        norm_gain = 1.0
        dx = dy = 0

        # 1) ref 기반 정규화 + ref 위치 추적(Δ 계산)
        if ref is not None:
            ref_id = ref["id"]
            rx, ry, rw, rh = ref["x"], ref["y"], ref["w"], ref["h"]

            # template은 "정규화 전 원본"에서 확보
            ref_crop_raw = self.roi_mgr.crop(frame_gray8, ref_id)
            
            if self.tracker.template is None:
                self.tracker.set_template(ref_crop_raw)

            # 정규화
            if ref_crop_raw is not None and ref_crop_raw.size > 0:
                frame_gray8, norm_gain = normalize_by_roi(frame_gray8, ref_crop_raw, target_mean=50.0)

            # ref만 추적해서 Δ 계산(정규화된 프레임에서)
            nrx, nry, _, _ = self.tracker.track(frame_gray8, rx, ry, rw, rh)
            dx, dy = int(nrx - rx), int(nry - ry)

        # 2) 모든 ROI는 Δ만 적용해서 crop (안정)
        H, W = frame_gray8.shape[:2]

        for roi in getattr(self.roi_mgr, "rois", []):
            roi_id = roi.get("id")
            key = str(roi_id)

            x, y, w, h = int(roi["x"]), int(roi["y"]), int(roi["w"]), int(roi["h"])
            x, y = x + dx, y + dy

            # clamp
            x = max(0, min(x, W - 1))
            y = max(0, min(y, H - 1))
            x2 = max(1, min(W, x + w))
            y2 = max(1, min(H, y + h))

            crop = frame_gray8[y:y2, x:x2]
            if crop is None or crop.size == 0:
                results[key] = ROIResult(roi_id=roi_id, ok=False, reason="EMPTY_CROP", metrics={})
                continue

            # cfg = get_roi_cfg(self.recipe, roi_id)
            # ok, metrics, reason = run_analyzer(crop, cfg)

            # === 분석 및 mean+score 기반 판정 통합 ===
            cfg = get_roi_cfg(self.recipe, roi_id)

            # 기존 analyzer 호출 (유지하되 metrics를 확장)
            ok, metrics, reason = run_analyzer(crop, cfg)

            # ensure metrics is a dict
            if metrics is None:
                metrics = {}

            # --- ensure mean exists for ALL ROIs ---
            mean_raw = float(np.mean(crop))
            metrics["mean_raw"] = mean_raw
            metrics["mean"] = self.mean_filter.update(mean_raw)

            # compute score (texture/edge) and attach
            try:
                score = combined_score(crop)
            except Exception:
                score = 0.0
            metrics["score"] = float(score)

            # recipe thresholds (default 및 ROI override)
            default_min = float(self.recipe.get("default", {}).get("min_mean", 0.0))
            default_max = float(self.recipe.get("default", {}).get("max_mean", 255.0))
            default_score_thresh = float(self.recipe.get("default", {}).get("score_threshold", 0.25))

            over = self.recipe.get("overrides", {}).get(cfg.get("name", f"ROI{roi_id}"), {})
            # allow override by ROI name or ROI{ID}
            if not over:
                # try by explicit ROI key (e.g. "ROI1")
                over = self.recipe.get("overrides", {}).get(f"ROI{roi_id}", {})

            min_mean = float(over.get("min_mean", cfg.get("min_mean", default_min)))
            max_mean = float(over.get("max_mean", cfg.get("max_mean", default_max)))
            score_thresh = float(over.get("score_threshold", default_score_thresh))

            # after metrics updated with 'mean' and 'score' etc.
            mean_val = float(metrics.get("mean", metrics.get("mean_raw", np.mean(crop))))
            mean_ok = (min_mean <= mean_val <= max_mean)
            score_ok = (float(metrics.get("score", 0.0)) >= float(score_thresh))

            roi_type = (cfg.get("type") or "").strip().lower()

            # --- final decision by ROI type ---
            if roi_type == "mean_threshold":
                final_ok = bool(mean_ok)
                reason = "OK" if final_ok else ("LOW_MEAN" if mean_val < min_mean else "HIGH_MEAN")

            elif roi_type == "score_threshold":
                final_ok = bool(score_ok)
                reason = "OK" if final_ok else "LOW_SCORE"

            else:
                # default: analyzer only
                final_ok = bool(ok)
                reason = "OK" if final_ok else (reason or "FAIL")

            metrics["norm_gain"] = float(norm_gain)
            metrics["dx"] = dx
            metrics["dy"] = dy

            results[key] = ROIResult(roi_id=roi_id, ok=final_ok, reason=reason, metrics=metrics)

            # --- overall decision by recipe ---
            decision = (self.recipe.get("decision") or {})
            mode = (decision.get("mode") or "any_fail_is_ng").strip().lower()

            oks = [r.ok for r in results.values()]
            if not oks:
                overall_ok = False
            else:
                if mode == "any_fail_is_ng":
                    overall_ok = all(oks)
                    print("[DBG] overall decision by recipe : any_fail_is_ng")

                elif mode == "majority_ok":
                    overall_ok = (sum(1 for v in oks if v) >= (len(oks) / 2))
                    print("[DBG] overall decision by recipe : majority_ok")

                elif mode == "allow_fail_count":
                    max_fail = int(decision.get("max_fail", 0))
                    fail_cnt = sum(1 for v in oks if not v)
                    overall_ok = (fail_cnt <= max_fail)
                    print("[DBG] overall decision by recipe : allow_fail_count")

                else:
                    # fallback
                    overall_ok = all(oks)
                    print("[DBG] overall decision by recipe : fallback")

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
                    "metrics": v.metrics,
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
        target_mean = 50.0
        margin = 10.0
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
            overrides[f"ROI{roi_id}"] = {
                "type": "mean_threshold",
                "min_mean": float(mn),
                "max_mean": float(mx),
            }

        recipe = {
            "default": {"type": "mean_threshold", "min_mean": 0.0, "max_mean": 255.0},
            "overrides": overrides,
            "decision": {"mode": "any_fail_is_ng"},
        }

        save_recipe(save_path, recipe)
        self.recipe = recipe  # 즉시 반영
        return recipe
    
    def reset_tracker_template(self):
        self.tracker.template = None
        # print("[RESET] tracker template cleared")

