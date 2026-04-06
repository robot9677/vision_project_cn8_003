import os
import time
import cv2
import numpy as np

from inspection.engine.result_model import ROIResult
from .inspection_runner import _empty_align_result

def _get_first_center_abs(roi, metrics):
    if not isinstance(metrics, dict):
        return None

    centers = metrics.get("centers") or []
    if centers:
        c = centers[0]
        x = float(roi.get("x", 0)) + float(c.get("x", 0))
        y = float(roi.get("y", 0)) + float(c.get("y", 0))
        return x, y

    circles = metrics.get("circles") or []
    if circles:
        c = circles[0]
        if isinstance(c, dict):
            x = float(roi.get("x", 0)) + float(c.get("x", 0))
            y = float(roi.get("y", 0)) + float(c.get("y", 0))
        else:
            x = float(roi.get("x", 0)) + float(c[0])
            y = float(roi.get("y", 0)) + float(c[1])
        return x, y

    return None


def _apply_roi_distance_links(*, results, rois, recipe):
    print("[DBG ROI DIST ENTRY]")
    roi_map = {int(r.get("id")): r for r in (rois or []) if r.get("id") is not None}
    link_cfgs = (recipe or {}).get("roi_distance_links") or []

    for item in link_cfgs:
        from_roi_id = int(item.get("from_roi_id", -1))
        to_roi_id = int(item.get("to_roi_id", -1))

        from_res = results.get(str(from_roi_id))
        to_res = results.get(str(to_roi_id))
        from_roi = roi_map.get(from_roi_id)
        to_roi = roi_map.get(to_roi_id)

        if from_res is None or to_res is None or from_roi is None or to_roi is None:
            continue

        from_metrics = getattr(from_res, "metrics", None)
        to_metrics = getattr(to_res, "metrics", None)

        if not isinstance(from_metrics, dict) or not isinstance(to_metrics, dict):
            continue

        p1 = _get_first_center_abs(from_roi, from_metrics)
        p2 = _get_first_center_abs(to_roi, to_metrics)

        if p1 is None or p2 is None:
            continue

        x1, y1 = p1
        x2, y2 = p2

        dx = float(x1) - float(x2)
        dy = float(y1) - float(y2)
        dist_px = float((dx * dx + dy * dy) ** 0.5)

        link = {
            "from_roi_id": int(from_roi_id),
            "to_roi_id": int(to_roi_id),
            "x1": float(x1),
            "y1": float(y1),
            "x2": float(x2),
            "y2": float(y2),
            "distance_px": dist_px,
        }

        mm_per_px = from_metrics.get("mm_per_px", None)
        if mm_per_px is None:
            mm_per_px = to_metrics.get("mm_per_px", None)

        if mm_per_px is not None:
            link["distance_mm"] = float(dist_px) * float(mm_per_px)

        from_metrics.setdefault("roi_distance_links", []).append(link)

        print(
            f"[DBG ROI DIST] "
            f"R{from_roi_id}->R{to_roi_id} "
            f"px={dist_px:.1f} "
            f"mm={link.get('distance_mm', None)} "
            f"cnt={len(from_metrics.get('roi_distance_links', []))}"
        )

def process_all_rois(
    *,
    inspector,
    frame_gray8,
    align_result,
    norm_gain,
    trk_score,
    auto_mode=False,
):
    results = {}

    if align_result is None:
        align_result = _empty_align_result()

    anchor_roi_ids = {int(a.get("roi_id", 0)) for a in getattr(inspector.aligner, "_anchors", [])}

    profile_rois = (inspector.runtime_cfg.get("_product_profile", {}) or {}).get("rois") or []
    roi_types = {
        int(r.get("id")): str(r.get("type", "")).strip().lower()
        for r in profile_rois
        if r.get("id") is not None
    }

    use_explicit_inspections = inspector.recipe is not None and "inspections" in inspector.recipe

    for roi in getattr(inspector.roi_mgr, "rois", []):
        roi_id = int(roi.get("id"))
        key = str(roi_id)

        roi_type = roi_types.get(roi_id, "")
        roi_has_job = False
        for item in (inspector.recipe.get("inspections") or []):
            if int(item.get("roi_id", -1)) == roi_id and bool(item.get("enabled", True)):
                roi_has_job = True
                break

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

        crop = inspector._crop_rotated(frame_gray8, roi, dx=roi_dx, dy=roi_dy, dangle=roi_dangle)

        if crop is None or crop.size == 0:
            results[key] = ROIResult(roi_id=roi_id, ok=False, reason="EMPTY_CROP", metrics={})
            continue

        inspection_cfgs = []
        for item in (inspector.recipe.get("inspections") or []):
            if int(item.get("roi_id", -1)) == roi_id and bool(item.get("enabled", True)):
                inspection_cfgs.append(item)

        job_results = []
        merged_metrics = {}
        final_ok = True
        final_reason = "OK"
        last_metrics = {}

        for cfg in inspection_cfgs:
            cfg = dict(cfg)
            cfg["product_profile"] = inspector.runtime_cfg.get("_product_profile", {}) or {}
            
            job_ok, metrics, job_reason, job_type = inspector._run_inspection_job(
                crop=crop,
                cfg=cfg,
                recipe_default=inspector.recipe.get("default", {}),
                runtime_cfg=inspector.runtime_cfg,
                mean_filter=inspector._get_mean_filter(roi_id),
                norm_gain=norm_gain,
                roi_dx=roi_dx,
                roi_dy=roi_dy,
                roi_dangle=roi_dangle,
                pose=pose,
                trk_score=trk_score,
            )

            if "tools" in cfg and cfg.get("tools"):
                if not auto_mode and roi_id == 1:
                    dbg_dir = os.path.join(inspector.logs_root, "_dbg")
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
                    "type": job_type,
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

        reason = final_reason if not final_ok else "OK"

        metrics = dict(merged_metrics)
        metrics["_inspections"] = job_results
        if isinstance(last_metrics, dict) and last_metrics.get("_last_image") is not None:
            metrics["_last_image"] = last_metrics.get("_last_image")

        inspector._show_debug_view(
            roi_id=roi_id,
            raw_crop=crop,
            last_img=metrics.get("_last_image"),
        )

        if roi_id not in anchor_roi_ids:
            b_ok, b_reason = inspector._check_baseline(roi_id, metrics)
            if not b_ok:
                final_ok = False
                reason = b_reason

        results[key] = ROIResult(roi_id=roi_id, ok=final_ok, reason=reason, metrics=metrics)

        if not auto_mode and inspector.runtime_cfg.get("debug_log", True):
            dbg_path = os.path.join(inspector.logs_root, f"roi{roi_id}_last.png")
            last_img = metrics.get("_last_image")
            if not auto_mode and last_img is not None:
                cv2.imwrite(dbg_path, last_img)
                print(f"[DBG SAVE] {dbg_path}")

            if metrics.get("_tool_steps") is not None:
                print(f"[DBG TOOLS ROI{roi_id}] {metrics.get('_tool_steps')}")

    print("[DBG ROI DIST CALL]")
    _apply_roi_distance_links(
        results=results,
        rois=getattr(inspector.roi_mgr, "rois", []),
        recipe=inspector.recipe or {},
    )

    return results