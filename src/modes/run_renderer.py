import os
import time

from ui import overlay_clean as overlay
from inspection.logger import save_snapshot, save_template_copy


def roi_mgr_to_list(roi_mgr):
    return [
        {
            "id": r.get("id"),
            "label": r.get("name"),
            "rect": (
                int(r.get("x", 0)),
                int(r.get("y", 0)),
                int(r.get("w", 0)),
                int(r.get("h", 0)),
            ),
            "angle": float(r.get("angle", 0.0)),
        }
        for r in getattr(roi_mgr, "rois", [])
    ]


def draw_roi_overlay(vis, moved, last_results, roi_label_pos, show_metrics=False):
    rois = []

    for mr in moved:
        rid = str(mr.get("id"))
        lr = last_results.get(rid) if last_results else None

        rois.append({
            "id": int(mr.get("id")),
            "label": mr.get("name", f"ROI{rid}"),
            "rect": (
                int(mr.get("x", 0)),
                int(mr.get("y", 0)),
                int(mr.get("w", 0)),
                int(mr.get("h", 0)),
            ),
            "angle": float(mr.get("angle", 0.0)),
        })

    roi_results = None
    if last_results:
        roi_results = {str(k): v for k, v in last_results.items()}

    overlay.draw_rois(
        vis,
        rois=rois,
        active_id=None,
        roi_results=roi_results,
        compact=True,
        show_metrics=show_metrics,
    )


def draw_run_tracking(
    vis,
    frame_gray8,
    *,
    runtime_cfg,
    product_profile,
    state,
    roi_mgr,
    inspector,
    stabilizer,
    data_dir,
    snapshot_cooldown,
    snapshot_keep,
    prune_snapshots,
    roi_label_pos,
    show_metrics=False,
):
    if (not runtime_cfg.get("enable_tracker", True)) or (not product_profile["modules"].get("tracker", True)):
        state.tracking_stable = False
        state.stable_frame_count = 0
        overlay.draw_rois(vis, rois=roi_mgr_to_list(roi_mgr), active_id=roi_mgr.selected_id, roi_results=state.last_results, compact=True)
        return

    aligner = getattr(inspector, "aligner", None)
    moved = []

    try:
        if aligner is not None and frame_gray8 is not None:
            rois_src = list(getattr(roi_mgr, "rois", []))

            align_result = aligner.estimate(frame_gray8, roi_mgr)
            moved = aligner.apply_to_rois(rois_src, align_result)

            any_ok = any(bool(a.get("ok")) for a in (align_result.get("anchors") or []))
            smoothed, stable = stabilizer.update(moved)

            state.tracking_stable = bool(any_ok)
            if state.tracking_stable:
                state.stable_frame_count += 1
            else:
                state.stable_frame_count = 0

            state.status = "RUN MODE (stable)" if state.tracking_stable else "RUN MODE (tracking...)"

            if state.tracking_stable and (time.time() - state.last_snapshot_time) > snapshot_cooldown:
                log_dir = os.path.join(data_dir, "logs", "snapshots")
                os.makedirs(log_dir, exist_ok=True)

                for mr in smoothed:
                    roi_for_save = {
                        "id": mr["id"],
                        "x": int(round(mr["x"])),
                        "y": int(round(mr["y"])),
                        "w": int(mr["w"]),
                        "h": int(mr["h"]),
                    }
                    save_snapshot(log_dir, frame_gray8, roi_for_save, prefix="stable")
                    prune_snapshots(log_dir, snapshot_keep)

                tracker = getattr(aligner, "primary_tracker", None)
                if tracker is not None and getattr(tracker, "template", None) is not None:
                    save_template_copy(log_dir, tracker.template)

                state.last_snapshot_time = time.time()

            draw_roi_overlay(vis, smoothed, state.last_results, roi_label_pos, show_metrics=show_metrics)
        else:
            overlay.draw_rois(vis, rois=roi_mgr_to_list(roi_mgr), active_id=roi_mgr.selected_id, roi_results=state.last_results, compact=True)
            state.tracking_stable = False
            state.stable_frame_count = 0

    except Exception as e:
        print("[DBG] run-mode tracker overlay exception:", e)
        overlay.draw_rois(vis,rois=rois,active_id=None,roi_results=roi_results,compact=True,show_metrics=show_metrics)
        state.tracking_stable = False
        state.stable_frame_count = 0