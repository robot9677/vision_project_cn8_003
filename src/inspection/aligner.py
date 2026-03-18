import os
from typing import Any, Dict, List, Optional

import cv2

from .roi_tracker import ROITracker


class MultiAnchorAligner:
    """
    Generalized 0..N anchor aligner.

    - 0 anchors  -> fixed ROI fallback
    - 1 anchor   -> global pose align
    - N anchors  -> per-group pose align

    Notes:
    - This version keeps the existing template-match tracker engine.
    - Each anchor owns one ROITracker and one template.
    - Targets can be 'all' or a list of ROI ids.
    """

    def __init__(
        self,
        runtime_cfg: Optional[Dict[str, Any]] = None,
        product_profile: Optional[Dict[str, Any]] = None,
        project_root: Optional[str] = None,
    ):
        self.runtime_cfg = runtime_cfg or {}
        self.product_profile = product_profile or {}
        self.project_root = project_root
        self._anchors: List[Dict[str, Any]] = []
        self.primary_tracker: Optional[ROITracker] = None
        self.refresh_config()

    def refresh_config(self):
        self._anchors = []
        align_cfg = self._get_align_cfg()
        anchors_cfg = align_cfg.get("anchors") or []

        if not anchors_cfg:
            anchors_cfg = [
                {
                    "id": "anchor_main",
                    "roi_id": 1,
                    "enabled": True,
                    "targets": "all",
                    "template_path": "data/roi/align_template.png",
                    "template_source": "file_or_runtime",
                }
            ]

        default_search_margin = int(self.runtime_cfg.get("tracker_search_margin", 80))
        default_thr = float(self.runtime_cfg.get("tracker_thr", 0.70))
        default_reacquire_margin = int(self.runtime_cfg.get("tracker_reacquire_margin", 220))
        default_reacquire_scale = float(self.runtime_cfg.get("tracker_reacquire_scale", 0.5))
        default_angle_range = float(self.runtime_cfg.get("tracker_angle_range", 4.0))
        default_angle_step = float(self.runtime_cfg.get("tracker_angle_step", 1.0))

        for idx, raw in enumerate(anchors_cfg):
            if not isinstance(raw, dict):
                continue

            roi_id = int(raw.get("roi_id", 1))
            anchor_id = str(raw.get("id") or f"anchor_{roi_id}_{idx + 1}")
            enabled = bool(raw.get("enabled", True))
            targets = raw.get("targets", "all")

            if isinstance(targets, list):
                targets = [int(v) for v in targets]
            elif targets != "all":
                targets = "all"

            tracker = ROITracker(
                search_margin=int(raw.get("search_margin", default_search_margin)),
                thr=float(raw.get("thr", default_thr)),
                reacquire_margin=int(raw.get("reacquire_margin", default_reacquire_margin)),
                reacquire_scale=float(raw.get("reacquire_scale", default_reacquire_scale)),
            )

            entry = {
                "id": anchor_id,
                "roi_id": roi_id,
                "enabled": enabled,
                "targets": targets,
                "template_path": raw.get("template_path"),
                "template_source": str(raw.get("template_source", "file_or_runtime")),
                "angle_range": float(raw.get("angle_range", default_angle_range)),
                "angle_step": float(raw.get("angle_step", default_angle_step)),
                "tracker": tracker,
            }
            self._anchors.append(entry)

        self.primary_tracker = self._anchors[0]["tracker"] if self._anchors else None

    def _get_align_cfg(self) -> Dict[str, Any]:
        from_profile = self.product_profile.get("align")
        if isinstance(from_profile, dict):
            return from_profile

        from_runtime = self.runtime_cfg.get("align")
        if isinstance(from_runtime, dict):
            return from_runtime

        return {
            "enabled": True,
            "min_score": 0.85,
            "fallback_mode": "fixed_roi",
            "anchors": [],
        }

    def _get_min_score(self) -> float:
        align_cfg = self._get_align_cfg()
        return float(align_cfg.get("min_score", 0.85))

    def _get_fallback_mode(self) -> str:
        align_cfg = self._get_align_cfg()
        return str(align_cfg.get("fallback_mode", "fixed_roi"))

    def is_enabled(self) -> bool:
        align_cfg = self._get_align_cfg()
        if not bool(align_cfg.get("enabled", True)):
            return False
        return bool(self.runtime_cfg.get("enable_tracker", True))

    def reset_templates(self):
        for a in self._anchors:
            a["tracker"].set_template(None)

    def _resolve_template_path(self, raw_path: Optional[str]) -> Optional[str]:
        if not raw_path:
            return None
        raw_path = str(raw_path)
        if os.path.isabs(raw_path):
            return raw_path
        if self.project_root:
            return os.path.join(self.project_root, raw_path)
        return raw_path

    def load_templates_from_disk(self) -> int:
        loaded = 0
        for a in self._anchors:
            path = self._resolve_template_path(a.get("template_path"))
            if not path or not os.path.exists(path):
                continue

            tpl = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if tpl is None or tpl.size == 0:
                continue

            a["tracker"].set_template(tpl)
            loaded += 1

        return loaded

    def ensure_runtime_templates(self, frame_gray8, roi_mgr):
        if frame_gray8 is None or roi_mgr is None:
            return

        for a in self._anchors:
            tracker = a["tracker"]
            if tracker.template is not None:
                continue

            src_mode = a.get("template_source", "file_or_runtime")
            if src_mode not in ("roi_runtime", "file_or_runtime", "auto"):
                continue

            crop = roi_mgr.crop(frame_gray8, a["roi_id"])
            if crop is not None and crop.size > 0:
                tracker.set_template(crop)

    def _make_fallback_pose(self, all_roi_ids: List[int], reason: str) -> Dict[str, Any]:
        per_roi = {}
        for rid in all_roi_ids:
            per_roi[int(rid)] = {
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": 0.0,
                "anchor_id": None,
                "fallback": True,
                "reason": reason,
            }

        return {
            "enabled": self.is_enabled(),
            "anchors": [],
            "per_roi": per_roi,
            "global": {
                "ok": False,
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": 0.0,
                "fallback": True,
                "reason": reason,
            },
            "fallback_mode": self._get_fallback_mode(),
        }

    def estimate(self, frame_gray8, roi_mgr) -> Dict[str, Any]:
        result = {
            "enabled": self.is_enabled(),
            "anchors": [],
            "per_roi": {},
            "global": {
                "ok": False,
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": 0.0,
                "fallback": False,
                "reason": "",
            },
            "fallback_mode": self._get_fallback_mode(),
        }

        if (not result["enabled"]) or frame_gray8 is None or roi_mgr is None:
            return result

        rois = list(getattr(roi_mgr, "rois", []))
        all_roi_ids = [int(r.get("id")) for r in rois]
        self.ensure_runtime_templates(frame_gray8, roi_mgr)

        min_score = self._get_min_score()
        best_global = None
        any_success = False

        for a in self._anchors:
            if not a.get("enabled", True):
                continue

            roi = roi_mgr.get(a["roi_id"])
            tracker = a["tracker"]

            if roi is None or tracker.template is None:
                result["anchors"].append({
                    "id": a["id"],
                    "roi_id": a["roi_id"],
                    "ok": False,
                    "dx": 0,
                    "dy": 0,
                    "dangle": 0.0,
                    "score": 0.0,
                    "reason": "NO_TEMPLATE_OR_ROI",
                })
                continue

            rx = int(roi.get("x", 0))
            ry = int(roi.get("y", 0))
            rw = int(roi.get("w", 0))
            rh = int(roi.get("h", 0))
            ra = float(roi.get("angle", 0.0))

            nrx, nry, _, _, na, score = tracker.track_pose(
                frame_gray8,
                rx, ry, rw, rh,
                angle=ra,
                angle_range=float(a.get("angle_range", 4.0)),
                angle_step=float(a.get("angle_step", 1.0)),
            )

            dx = int(nrx - rx)
            dy = int(nry - ry)
            dangle = float(na - ra)
            score = float(score)
            ok = score >= min_score

            anchor_pose = {
                "id": a["id"],
                "roi_id": a["roi_id"],
                "ok": bool(ok),
                "dx": dx,
                "dy": dy,
                "dangle": dangle,
                "score": score,
                "reason": "OK" if ok else "LOW_SCORE",
            }
            result["anchors"].append(anchor_pose)

            if not ok:
                print(f"[DBG ALIGN] {a['id']} ok=False (LOW_SCORE {score:.3f} < {min_score:.3f})")
                continue

            print(f"[DBG ALIGN] {a['id']} ok=True dx={dx} dy={dy} da={dangle:.2f} sc={score:.3f}")

            any_success = True
            targets = all_roi_ids if a.get("targets") == "all" else list(a.get("targets") or [])

            for rid in targets:
                prev = result["per_roi"].get(int(rid))
                if prev is None or score > float(prev.get("score", -1.0)):
                    result["per_roi"][int(rid)] = {
                        "dx": dx,
                        "dy": dy,
                        "dangle": dangle,
                        "score": score,
                        "anchor_id": a["id"],
                        "fallback": False,
                        "reason": "OK",
                    }

            if best_global is None or score > float(best_global.get("score", -1.0)):
                best_global = anchor_pose

        if not any_success:
            fallback_mode = self._get_fallback_mode()

            if fallback_mode == "fixed_roi":
                print("[FALLBACK] fixed_roi")
                return self._make_fallback_pose(all_roi_ids, "LOW_SCORE_ALL")

            print(f"[FALLBACK] unsupported fallback_mode={fallback_mode}, use fixed_roi")
            return self._make_fallback_pose(all_roi_ids, "LOW_SCORE_ALL")

        for rid in all_roi_ids:
            result["per_roi"].setdefault(int(rid), {
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": 0.0,
                "anchor_id": None,
                "fallback": True,
                "reason": "UNASSIGNED_USE_FIXED",
            })

        if best_global is not None:
            result["global"] = {
                "ok": True,
                "dx": int(best_global.get("dx", 0)),
                "dy": int(best_global.get("dy", 0)),
                "dangle": float(best_global.get("dangle", 0.0)),
                "score": float(best_global.get("score", 0.0)),
                "fallback": False,
                "reason": "OK",
            }

        return result

    def apply_to_rois(self, rois: List[Dict[str, Any]], align_result: Dict[str, Any]) -> List[Dict[str, Any]]:
        moved = []
        per_roi = align_result.get("per_roi") or {}

        for r in rois:
            rid = int(r.get("id", 0))
            pose = per_roi.get(rid, {})

            moved.append({
                "id": r.get("id"),
                "name": r.get("name", ""),
                "x": int(r.get("x", 0) + pose.get("dx", 0)),
                "y": int(r.get("y", 0) + pose.get("dy", 0)),
                "w": int(r.get("w", 0)),
                "h": int(r.get("h", 0)),
                "angle": float(r.get("angle", 0.0) + pose.get("dangle", 0.0)),
            })

        return moved