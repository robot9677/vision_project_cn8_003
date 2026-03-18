import os
from typing import Any, Dict, List, Optional

import cv2

from .roi_tracker import ROITracker


class MultiAnchorAligner:
    """
    0..N anchor aligner with hold/grace tracking state.

    핵심:
    - last pose 유지
    - grace_frames 동안은 low score여도 즉시 fixed_roi 복귀 안 함
    - 다음 검색 중심은 원래 ROI가 아니라 last pose 반영 위치 기준
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
        self._last_global_state = None

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
                search_margin_x=int(raw.get("search_margin_x", raw.get("search_margin", default_search_margin))),
                search_margin_y=int(raw.get("search_margin_y", raw.get("search_margin", default_search_margin))),
                thr=float(raw.get("thr", default_thr)),
                reacquire_margin=int(raw.get("reacquire_margin", default_reacquire_margin)),
                reacquire_scale=float(raw.get("reacquire_scale", default_reacquire_scale)),
            )

            tracker.wide_reacquire_margin_x = int(raw.get("wide_reacquire_margin_x", tracker.search_margin_x * 3))
            tracker.wide_reacquire_margin_y = int(raw.get("wide_reacquire_margin_y", tracker.search_margin_y * 3))

            entry = {
                "id": anchor_id,
                "roi_id": roi_id,
                "enabled": enabled,
                "targets": targets,
                "template_path": raw.get("template_path"),
                "template_source": str(raw.get("template_source", "file_or_runtime")),
                "angle_range": float(raw.get("angle_range", 0.0)),
                "angle_step": float(raw.get("angle_step", 1.0)),
                "enable_rotation": bool(raw.get("enable_rotation", False)),
                "max_abs_angle": float(raw.get("max_abs_angle", 8.0)),
                "tracker": tracker,
                "log_state": None,
                "log_score_band": None,
                "last_output_dx": 0,
                "last_output_dy": 0,
                "last_output_dangle": 0.0,
                "stable_count": 0,

                # tracking state
                "last_pose": {
                    "dx": 0,
                    "dy": 0,
                    "dangle": 0.0,
                    "score": 0.0,
                },
                "fail_count": 0,
                "has_lock": False,
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
            "min_score": 0.82,
            "grace_frames": 8,
            "fallback_mode": "fixed_roi",
            "anchors": [],
        }

    def _get_min_score(self) -> float:
        return float(self._get_align_cfg().get("min_score", 0.82))

    def _get_grace_frames(self) -> int:
        return int(self._get_align_cfg().get("grace_frames", 8))

    def _get_fallback_mode(self) -> str:
        return str(self._get_align_cfg().get("fallback_mode", "fixed_roi"))

    def is_enabled(self) -> bool:
        align_cfg = self._get_align_cfg()
        if not bool(align_cfg.get("enabled", True)):
            return False
        return bool(self.runtime_cfg.get("enable_tracker", True))

    def reset_templates(self):
        for a in self._anchors:
            a["tracker"].set_template(None)
            a["last_pose"] = {"dx": 0, "dy": 0, "dangle": 0.0, "score": 0.0}
            a["fail_count"] = 0
            a["has_lock"] = False
            a["last_output_dx"] = 0
            a["last_output_dy"] = 0
            a["last_output_dangle"] = 0.0
            a["log_state"] = None
            a["log_score_band"] = None

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
            # 항상 최신 ROI로 템플릿 갱신 (디버깅용)
            crop = roi_mgr.crop(frame_gray8, a["roi_id"])
            if crop is not None and crop.size > 0:
                tracker.set_template(crop)

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

    def _make_hold_pose(self, anchor_id: str, pose: Dict[str, Any], all_roi_ids: List[int], reason: str) -> Dict[str, Any]:
        per_roi = {}
        for rid in all_roi_ids:
            per_roi[int(rid)] = {
                "dx": int(pose.get("dx", 0)),
                "dy": int(pose.get("dy", 0)),
                "dangle": float(pose.get("dangle", 0.0)),
                "score": float(pose.get("score", 0.0)),
                "anchor_id": anchor_id,
                "fallback": False,
                "reason": reason,
            }

        return {
            "enabled": self.is_enabled(),
            "anchors": [],
            "per_roi": per_roi,
            "global": {
                "ok": True,
                "dx": int(pose.get("dx", 0)),
                "dy": int(pose.get("dy", 0)),
                "dangle": float(pose.get("dangle", 0.0)),
                "score": float(pose.get("score", 0.0)),
                "fallback": False,
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
        grace_frames = self._get_grace_frames()
        best_global = None
        any_success = False
        any_hold = False

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

            base_x = int(roi.get("x", 0))
            base_y = int(roi.get("y", 0))
            base_w = int(roi.get("w", 0))
            base_h = int(roi.get("h", 0))
            base_a = float(roi.get("angle", 0.0))

            # 핵심: 직전 성공 pose를 적용한 위치를 다음 검색 중심으로 사용
            last_pose = a.get("last_pose", {})
            search_x = base_x + int(last_pose.get("dx", 0))
            search_y = base_y + int(last_pose.get("dy", 0))
            search_a = base_a + float(last_pose.get("dangle", 0.0))

            nrx, nry, _, _, na, score = tracker.track_pose(
                frame_gray8,
                search_x, search_y, base_w, base_h,
                angle=search_a,
                angle_range=float(a.get("angle_range", 0.0)),
                angle_step=float(a.get("angle_step", 1.0)),
                base_angle=base_a,
                enable_rotation=bool(a.get("enable_rotation", False)),
                max_abs_angle=float(a.get("max_abs_angle", 8.0)),
            )

            # base 기준 pose로 환산
            dx = int(nrx - base_x)
            dy = int(nry - base_y)
            dangle = float(na - base_a)
            score = float(score)

            ok = score >= min_score

            if ok:
                a["last_pose"] = {
                    "dx": dx,
                    "dy": dy,
                    "dangle": dangle,
                    "score": score,
                }
                a["fail_count"] = 0
                a["has_lock"] = True

                dx, dy, dangle = self._clamp_step(a, dx, dy, dangle)
                self._commit_output_pose(a, dx, dy, dangle)

                anchor_pose = {
                    "id": a["id"],
                    "roi_id": a["roi_id"],
                    "ok": True,
                    "dx": dx,
                    "dy": dy,
                    "dangle": dangle,
                    "score": score,
                    "reason": "OK",
                }
                result["anchors"].append(anchor_pose)
                self._last_global_state = "TRACKING"
                if self._should_log_anchor(a, "OK", dx, dy, score):
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
                continue

            # low score
            a["fail_count"] = int(a.get("fail_count", 0)) + 1
            fail_count = a["fail_count"]
            has_lock = bool(a.get("has_lock", False))

            # grace frames 동안은 마지막 성공 pose 유지
            if has_lock and fail_count <= grace_frames:
                hold_pose = a["last_pose"]
                hdx = int(hold_pose.get("dx", 0))
                hdy = int(hold_pose.get("dy", 0))
                hda = float(hold_pose.get("dangle", 0.0))
                hsc = float(hold_pose.get("score", 0.0))

                hdx, hdy, hda = self._clamp_step(a, hdx, hdy, hda)
                self._commit_output_pose(a, hdx, hdy, hda)

                result["anchors"].append({
                    "id": a["id"],
                    "roi_id": a["roi_id"],
                    "ok": True,
                    "dx": hdx,
                    "dy": hdy,
                    "dangle": hda,
                    "score": hsc,
                    "reason": f"HOLD({fail_count}/{grace_frames})",
                })

                self._last_global_state = "TRACKING"
                if fail_count == 1 and self._should_log_anchor(a, "HOLD", hdx, hdy, score):
                    print(
                        f"[DBG ALIGN] {a['id']} hold=True "
                        f"last_dx={hdx} last_dy={hdy} last_da={hda:.2f} "
                        f"low_sc={score:.3f} fail={fail_count}/{grace_frames}"
                    )
                any_hold = True
                targets = all_roi_ids if a.get("targets") == "all" else list(a.get("targets") or [])
                for rid in targets:
                    prev = result["per_roi"].get(int(rid))
                    if prev is None or hsc > float(prev.get("score", -1.0)):
                        result["per_roi"][int(rid)] = {
                            "dx": hdx,
                            "dy": hdy,
                            "dangle": hda,
                            "score": hsc,
                            "anchor_id": a["id"],
                            "fallback": False,
                            "reason": f"HOLD({fail_count}/{grace_frames})",
                        }

                if best_global is None or hsc > float(best_global.get("score", -1.0)):
                    best_global = {
                        "id": a["id"],
                        "roi_id": a["roi_id"],
                        "ok": True,
                        "dx": hdx,
                        "dy": hdy,
                        "dangle": hda,
                        "score": hsc,
                        "reason": f"HOLD({fail_count}/{grace_frames})",
                    }
                continue

            # grace 초과 시에만 fallback
            a["has_lock"] = False
            a["last_pose"] = {"dx": 0, "dy": 0, "dangle": 0.0, "score": 0.0}

            result["anchors"].append({
                "id": a["id"],
                "roi_id": a["roi_id"],
                "ok": False,
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": score,
                "reason": f"LOW_SCORE({fail_count}>{grace_frames})",
            })

        if not any_success and not any_hold:
            fallback_mode = self._get_fallback_mode()

            if fallback_mode == "hold":
                per_roi = {}
                for rid in all_roi_ids:
                    per_roi[int(rid)] = {
                        "dx": 0,
                        "dy": 0,
                        "dangle": 0.0,
                        "score": 0.0,
                        "anchor_id": None,
                        "fallback": False,
                        "reason": "HOLD_NO_SUCCESS",
                    }

                gdx = 0
                gdy = 0
                gda = 0.0
                gsc = 0.0
                gid = None

                for a in self._anchors:
                    if not a.get("enabled", True):
                        continue

                    adx = int(a.get("last_output_dx", 0))
                    ady = int(a.get("last_output_dy", 0))
                    ada = float(a.get("last_output_dangle", 0.0))
                    asc = float(a.get("last_pose", {}).get("score", 0.0))

                    targets = all_roi_ids if a.get("targets") == "all" else list(a.get("targets") or [])
                    for rid in targets:
                        per_roi[int(rid)] = {
                            "dx": adx,
                            "dy": ady,
                            "dangle": ada,
                            "score": asc,
                            "anchor_id": a.get("id"),
                            "fallback": False,
                            "reason": "HOLD_NO_SUCCESS",
                        }

                    if asc >= gsc:
                        gdx = adx
                        gdy = ady
                        gda = ada
                        gsc = asc
                        gid = a.get("id")

                result["per_roi"] = per_roi
                result["global"] = {
                    "ok": False,
                    "dx": gdx,
                    "dy": gdy,
                    "dangle": gda,
                    "score": gsc,
                    "fallback": False,
                    "reason": "HOLD_NO_SUCCESS",
                    "anchor_id": gid,
                }
                return result

            if self._last_global_state != "FALLBACK":
                print("[FALLBACK] fixed_roi")
            self._last_global_state = "FALLBACK"
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
                "reason": str(best_global.get("reason", "OK")),
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

    def _score_band(self, score: float) -> str:
        s = float(score)
        if s >= 0.95:
            return "S"
        if s >= 0.90:
            return "A"
        if s >= 0.85:
            return "B"
        if s >= 0.80:
            return "C"
        return "D"

    def _should_log_anchor(self, anchor: Dict[str, Any], state: str, dx: int, dy: int, score: float) -> bool:
        prev_state = anchor.get("log_state")
        prev_band = anchor.get("log_score_band")
        band = self._score_band(score)

        if prev_state != state or prev_band != band:
            anchor["log_state"] = state
            anchor["log_score_band"] = band
            return True

        return False

    def _clamp_step(self, anchor: Dict[str, Any], dx: int, dy: int, dangle: float):
        max_step_x = 18
        max_step_y = 18
        max_step_angle = 0.0

        prev_dx = int(anchor.get("last_output_dx", 0))
        prev_dy = int(anchor.get("last_output_dy", 0))
        prev_da = float(anchor.get("last_output_dangle", 0.0))

        ddx = dx - prev_dx
        ddy = dy - prev_dy
        dda = dangle - prev_da

        if ddx > max_step_x:
            dx = prev_dx + max_step_x
        elif ddx < -max_step_x:
            dx = prev_dx - max_step_x

        if ddy > max_step_y:
            dy = prev_dy + max_step_y
        elif ddy < -max_step_y:
            dy = prev_dy - max_step_y

        if dda > max_step_angle:
            dangle = prev_da + max_step_angle
        elif dda < -max_step_angle:
            dangle = prev_da - max_step_angle

        return int(dx), int(dy), float(dangle)

    def _commit_output_pose(self, anchor: Dict[str, Any], dx: int, dy: int, dangle: float):
        anchor["last_output_dx"] = int(dx)
        anchor["last_output_dy"] = int(dy)
        anchor["last_output_dangle"] = float(dangle)