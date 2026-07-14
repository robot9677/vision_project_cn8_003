import os
from typing import Any, Dict, List, Optional

import cv2

from app.app_paths import TEMPLATE_PATH, PROFILES_DIR
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
                    "template_path": TEMPLATE_PATH,
                    "template_source": "file_or_runtime",
                }
            ]

        default_search_margin = int(self.runtime_cfg.get("tracker_search_margin", 80))
        default_thr = float(self.runtime_cfg.get("tracker_thr", 0.70))
        default_reacquire_margin = int(self.runtime_cfg.get("tracker_reacquire_margin", 220))
        default_reacquire_scale = float(self.runtime_cfg.get("tracker_reacquire_scale", 0.5))

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
                runtime_cfg=(self.runtime_cfg.get("tracker", {}) if isinstance(self.runtime_cfg, dict) else {}),
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
        profile = str(
            self.product_profile.get("recipe_name")
            or (self.runtime_cfg.get("_product_profile") or {}).get("recipe_name")
            or ""
        ).strip()

        for index, anchor in enumerate(self._anchors):
            src_mode = anchor.get("template_source", "file_or_runtime")
            tracker = anchor["tracker"]

            if src_mode == "roi_runtime":
                continue

            # Reload must not leave a stale template from a previous profile.
            tracker.set_template(None)

            candidates = []
            configured_path = self._resolve_template_path(anchor.get("template_path"))
            if configured_path:
                candidates.append(configured_path)

            # The profile and generic fallback images represent the primary
            # anchor only. Additional anchors must use their configured paths.
            if index == 0 and profile:
                candidates.append(
                    os.path.join(PROFILES_DIR, f"align_template_{profile}.png")
                )
            if index == 0:
                candidates.append(TEMPLATE_PATH)

            path = next(
                (candidate for candidate in candidates if candidate and os.path.exists(candidate)),
                None,
            )
            if path is None:
                continue

            template = cv2.imread(path, cv2.IMREAD_GRAYSCALE)
            if template is None or template.size == 0:
                continue

            tracker.set_template(template)
            loaded += 1

        return loaded

    def ensure_runtime_templates(self, frame_gray8, roi_mgr):
        if frame_gray8 is None or roi_mgr is None:
            return

        for a in self._anchors:
            tracker = a["tracker"]

            src_mode = a.get("template_source", "file_or_runtime")
            if src_mode not in ("roi_runtime", "file_or_runtime", "auto"):
                continue

            # 템플릿이 없을 때만 1회 생성
            if tracker.template is not None:
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

        align_cfg = self._get_align_cfg()
        min_score = self._get_min_score()
        grace_frames = max(0, self._get_grace_frames())
        hold_min_score = float(align_cfg.get("hold_min_score", min_score - 0.12))
        smooth_cfg = (
            align_cfg.get("smooth", {})
            if isinstance(align_cfg.get("smooth", {}), dict)
            else {}
        )
        max_step_x = int(smooth_cfg.get("max_step_x", 9999))
        max_step_y = int(smooth_cfg.get("max_step_y", 9999))
        max_step_angle = float(smooth_cfg.get("max_step_angle", 999.0))
        jump_guard_score = float(align_cfg.get("jump_guard_score", min_score + 0.08))

        best_global = None
        any_success = False
        any_hold = False

        for anchor in self._anchors:
            if not anchor.get("enabled", True):
                continue

            roi = roi_mgr.get(anchor["roi_id"])
            tracker = anchor["tracker"]

            if roi is None or tracker.template is None:
                result["anchors"].append({
                    "id": anchor["id"],
                    "roi_id": anchor["roi_id"],
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

            last_pose = anchor.get("last_pose", {})
            previous_fail_count = int(anchor.get("fail_count", 0))
            was_locked = bool(anchor.get("has_lock", False))

            # Search around the last accepted pose. During grace frames this
            # keeps reacquisition local instead of snapping back to fixed ROI.
            search_x = base_x + int(last_pose.get("dx", 0))
            search_y = base_y + int(last_pose.get("dy", 0))
            search_a = base_a + float(last_pose.get("dangle", 0.0))

            nrx, nry, _, _, na, score = tracker.track_pose(
                frame_gray8,
                search_x,
                search_y,
                base_w,
                base_h,
                angle=search_a,
                angle_range=float(anchor.get("angle_range", 0.0)),
                angle_step=float(anchor.get("angle_step", 1.0)),
                base_angle=base_a,
                enable_rotation=bool(anchor.get("enable_rotation", False)),
                max_abs_angle=float(anchor.get("max_abs_angle", 8.0)),
            )

            raw_dx = int(nrx - base_x)
            raw_dy = int(nry - base_y)
            raw_dangle = float(na - base_a)
            score = float(score)

            strong_match = score >= min_score
            weak_hold_match = was_locked and score >= hold_min_score
            match_ok = strong_match or weak_hold_match
            reject_reason = "LOW_SCORE"

            if match_ok:
                prev_dx = int(anchor.get("last_output_dx", 0))
                prev_dy = int(anchor.get("last_output_dy", 0))
                prev_angle = float(anchor.get("last_output_dangle", 0.0))

                jump_x = abs(raw_dx - prev_dx)
                jump_y = abs(raw_dy - prev_dy)
                jump_angle = abs(raw_dangle - prev_angle)

                suspicious_jump = (
                    jump_x > (max_step_x * 2)
                    or jump_y > (max_step_y * 2)
                    or jump_angle > (max_step_angle * 2.0)
                )

                step_reject = (
                    (jump_x > max_step_x or jump_y > max_step_y)
                    and score < (jump_guard_score + 0.03)
                )
                jump_reject = (
                    suspicious_jump
                    and previous_fail_count == 0
                    and score < jump_guard_score
                )

                if step_reject:
                    match_ok = False
                    reject_reason = "STEP_REJECT"
                elif jump_reject:
                    match_ok = False
                    reject_reason = "JUMP_REJECT"

            if match_ok:
                anchor["fail_count"] = 0
                anchor["has_lock"] = True

                dx, dy, dangle = self._clamp_step(
                    anchor, raw_dx, raw_dy, raw_dangle
                )
                self._commit_output_pose(anchor, dx, dy, dangle)

                anchor["last_pose"] = {
                    "dx": dx,
                    "dy": dy,
                    "dangle": dangle,
                    "score": score,
                }

                success_reason = "OK" if strong_match else "HOLD_SCORE"
                anchor_pose = self._append_success_result(
                    result=result,
                    all_roi_ids=all_roi_ids,
                    anchor=anchor,
                    dx=dx,
                    dy=dy,
                    dangle=dangle,
                    score=score,
                    reason=success_reason,
                )

                self._last_global_state = "TRACKING" if strong_match else "HOLD"
                log_state = "OK" if strong_match else "HOLD_SCORE"
                if self._should_log_anchor(anchor, log_state, dx, dy, score):
                    print(
                        f"[DBG ALIGN] {anchor['id']} reason={success_reason} "
                        f"dx={dx} dy={dy} da={dangle:.2f} sc={score:.3f}"
                    )

                any_success = True
                if best_global is None or score > float(best_global.get("score", -1.0)):
                    best_global = anchor_pose
                continue

            fail_count = previous_fail_count + 1
            anchor["fail_count"] = fail_count

            if was_locked and fail_count <= grace_frames:
                held_dx = int(last_pose.get("dx", anchor.get("last_output_dx", 0)))
                held_dy = int(last_pose.get("dy", anchor.get("last_output_dy", 0)))
                held_angle = float(
                    last_pose.get("dangle", anchor.get("last_output_dangle", 0.0))
                )

                hold_pose = self._append_hold_result(
                    result=result,
                    all_roi_ids=all_roi_ids,
                    anchor=anchor,
                    dx=held_dx,
                    dy=held_dy,
                    dangle=held_angle,
                    score=score,
                    fail_count=fail_count,
                    grace_frames=grace_frames,
                )
                any_hold = True
                self._last_global_state = "HOLD"

                if self._should_log_anchor(anchor, "HOLD", held_dx, held_dy, score):
                    print(
                        f"[DBG ALIGN] {anchor['id']} hold=True "
                        f"count={fail_count}/{grace_frames} sc={score:.3f}"
                    )

                if best_global is None or score > float(best_global.get("score", -1.0)):
                    best_global = hold_pose
                continue

            # Grace expired (or no previous lock): release the lock. The last
            # output remains available for fallback_mode=hold overlay only.
            anchor["has_lock"] = False
            anchor["last_pose"] = {
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": 0.0,
            }

            result["anchors"].append({
                "id": anchor["id"],
                "roi_id": anchor["roi_id"],
                "ok": False,
                "dx": 0,
                "dy": 0,
                "dangle": 0.0,
                "score": score,
                "reason": f"{reject_reason}({fail_count}>{grace_frames})",
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

                global_dx = 0
                global_dy = 0
                global_angle = 0.0
                global_score = 0.0
                global_anchor_id = None

                for anchor in self._anchors:
                    if not anchor.get("enabled", True):
                        continue

                    dx = int(anchor.get("last_output_dx", 0))
                    dy = int(anchor.get("last_output_dy", 0))
                    dangle = float(anchor.get("last_output_dangle", 0.0))
                    score = float(anchor.get("last_pose", {}).get("score", 0.0))

                    targets = (
                        all_roi_ids
                        if anchor.get("targets") == "all"
                        else list(anchor.get("targets") or [])
                    )
                    for rid in targets:
                        per_roi[int(rid)] = {
                            "dx": dx,
                            "dy": dy,
                            "dangle": dangle,
                            "score": score,
                            "anchor_id": anchor.get("id"),
                            "fallback": False,
                            "reason": "HOLD_NO_SUCCESS",
                        }

                    if score >= global_score:
                        global_dx = dx
                        global_dy = dy
                        global_angle = dangle
                        global_score = score
                        global_anchor_id = anchor.get("id")

                result["per_roi"] = per_roi
                result["global"] = {
                    "ok": False,
                    "dx": global_dx,
                    "dy": global_dy,
                    "dangle": global_angle,
                    "score": global_score,
                    "fallback": False,
                    "reason": "HOLD_NO_SUCCESS",
                    "anchor_id": global_anchor_id,
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
                "anchor_id": best_global.get("id"),
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

    def _append_success_result(
        self,
        result: Dict[str, Any],
        all_roi_ids: List[int],
        anchor: Dict[str, Any],
        dx: int,
        dy: int,
        dangle: float,
        score: float,
        reason: str = "OK",
    ):
        anchor_pose = {
            "id": anchor["id"],
            "roi_id": anchor["roi_id"],
            "ok": True,
            "dx": dx,
            "dy": dy,
            "dangle": dangle,
            "score": score,
            "reason": reason,
        }
        result["anchors"].append(anchor_pose)

        targets = all_roi_ids if anchor.get("targets") == "all" else list(anchor.get("targets") or [])
        for rid in targets:
            prev = result["per_roi"].get(int(rid))
            if prev is None or score > float(prev.get("score", -1.0)):
                result["per_roi"][int(rid)] = {
                    "dx": dx,
                    "dy": dy,
                    "dangle": dangle,
                    "score": score,
                    "anchor_id": anchor["id"],
                    "fallback": False,
                    "reason": reason,
                }

        return anchor_pose

    def _append_hold_result(
        self,
        result: Dict[str, Any],
        all_roi_ids: List[int],
        anchor: Dict[str, Any],
        dx: int,
        dy: int,
        dangle: float,
        score: float,
        fail_count: int,
        grace_frames: int,
    ):
        reason = f"HOLD({fail_count}/{grace_frames})"

        hold_pose = {
            "id": anchor["id"],
            "roi_id": anchor["roi_id"],
            "ok": True,
            "dx": dx,
            "dy": dy,
            "dangle": dangle,
            "score": score,
            "reason": reason,
        }
        result["anchors"].append(hold_pose)

        targets = all_roi_ids if anchor.get("targets") == "all" else list(anchor.get("targets") or [])
        for rid in targets:
            prev = result["per_roi"].get(int(rid))
            if prev is None or score > float(prev.get("score", -1.0)):
                result["per_roi"][int(rid)] = {
                    "dx": dx,
                    "dy": dy,
                    "dangle": dangle,
                    "score": score,
                    "anchor_id": anchor["id"],
                    "fallback": False,
                    "reason": reason,
                }

        return hold_pose
    
    def _clamp_step(self, anchor: Dict[str, Any], dx: int, dy: int, dangle: float):
        align_cfg = self._get_align_cfg()
        smooth_cfg = align_cfg.get("smooth", {}) if isinstance(align_cfg.get("smooth", {}), dict) else {}

        max_step_x = int(smooth_cfg.get("max_step_x", 9999))
        max_step_y = int(smooth_cfg.get("max_step_y", 9999))
        max_step_angle = float(smooth_cfg.get("max_step_angle", 999.0))

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