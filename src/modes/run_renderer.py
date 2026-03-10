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


def draw_roi_overlay(vis, moved, last_results, roi_label_pos):
    for mr in moved:
        rid = str(mr.get("id"))
        lr = last_results.get(rid) if last_results else None

        x = int(mr["x"])
        y = int(mr["y"])
        w = int(mr["w"])
        h = int(mr["h"])

        if lr is None:
            overlay.draw_rect(vis, (x, y), (x + w, y + h), color=(0, 200, 0), thickness=2)
            tx, ty = roi_label_pos(x, y, w, h)
            overlay.draw_text(vis, f"ROI{rid}", (tx, ty + 14), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")
            continue

        metrics = getattr(lr, "metrics", None) or {}
        ok_flag = bool(getattr(lr, "ok", False))
        reason = getattr(lr, "reason", "") or ""

        box_color = (0, 200, 0) if ok_flag else (0, 0, 200)
        overlay.draw_rois(vis,rois=[{"id": rid,"label": f"ROI{rid}","rect": (x, y, w, h),"angle": float(mr.get("angle",0.0))}],active_id=None,roi_results=last_results)

        line1 = f"ROI{rid} {'OK' if ok_flag else 'NG'}"
        if ok_flag:
            parts = []
            mean_v = metrics.get("mean", metrics.get("mean_raw"))
            score_v = metrics.get("score")
            wr = metrics.get("white_ratio")
            edge = metrics.get("edge_energy", metrics.get("lap_var", metrics.get("laplacian_var")))
            qr = metrics.get("qr_data")
            bc = metrics.get("blob_count")

            if mean_v is not None:
                parts.append(f"m:{float(mean_v):.1f}")
            if score_v is not None:
                parts.append(f"s:{float(score_v):.2f}")
            if wr is not None:
                parts.append(f"wr:{float(wr):.2f}")
            if edge is not None:
                parts.append(f"e:{float(edge):.1f}")
            if bc is not None:
                parts.append(f"bc:{int(bc)}")
            if qr:
                parts.append(f"qr:{str(qr)[:12]}")
            line2 = " ".join(parts[:3]) if parts else ""
        else:
            line2 = str(reason)[:24] if reason else "FAIL"

        tx, ty = roi_label_pos(x, y, w, h)
        if line2:
            overlay.draw_text(vis, line2, (tx, ty), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")
        overlay.draw_text(vis, line1, (tx, ty + 14), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")


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
):
    if (not runtime_cfg.get("enable_tracker", True)) or (not product_profile["modules"].get("tracker", True)):
        state.tracking_stable = False
        state.stable_frame_count = 0
        overlay.draw_rois(
            vis,
            rois=roi_mgr_to_list(roi_mgr),
            active_id=roi_mgr.selected_id,
            roi_results=state.last_results,
        )
        return

    tracker = getattr(inspector, "tracker", None)
    moved = []

    try:
        if tracker is not None and getattr(tracker, "template", None) is not None and frame_gray8 is not None:
            if hasattr(roi_mgr, "list"):
                rois_src = roi_mgr.list()
            elif hasattr(roi_mgr, "get_rois"):
                rois_src = roi_mgr.get_rois()
            else:
                rois_src = []

            for r in rois_src:
                x = int(r.get("x", 0))
                y = int(r.get("y", 0))
                w = int(r.get("w", 0))
                h = int(r.get("h", 0))

                try:
                    out = tracker.track(frame_gray8, x, y, w, h) if hasattr(tracker, "track") else None
                    if out is None:
                        nx, ny, nw, nh = x, y, w, h
                    elif isinstance(out, (list, tuple)) and len(out) == 4:
                        nx, ny, nw, nh = map(int, out)
                    elif isinstance(out, (list, tuple)) and len(out) == 2:
                        dx, dy = map(int, out)
                        nx, ny, nw, nh = x + dx, y + dy, w, h
                    elif isinstance(out, dict):
                        nx = int(out.get("x", x))
                        ny = int(out.get("y", y))
                        nw = int(out.get("w", w))
                        nh = int(out.get("h", h))
                    else:
                        nx, ny, nw, nh = x, y, w, h
                except Exception:
                    nx, ny, nw, nh = x, y, w, h

                moved.append({"id": r.get("id"), "name": r.get("name", ""), "x": nx, "y": ny, "w": nw, "h": nh})

            smoothed, stable = stabilizer.update(moved)

            state.tracking_stable = bool(stable)
            if stable:
                state.stable_frame_count += 1
            else:
                state.stable_frame_count = 0

            state.status = "RUN MODE (stable)" if stable else "RUN MODE (tracking...)"

            if stable and (time.time() - state.last_snapshot_time) > snapshot_cooldown:
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

                if tracker is not None and getattr(tracker, "template", None) is not None:
                    save_template_copy(log_dir, tracker.template)

                state.last_snapshot_time = time.time()

            draw_roi_overlay(vis, moved, state.last_results, roi_label_pos)
        else:
            overlay.draw_rois(
                vis,
                rois=roi_mgr_to_list(roi_mgr),
                active_id=roi_mgr.selected_id,
                roi_results=state.last_results,
            )
            state.tracking_stable = False
            state.stable_frame_count = 0

    except Exception as e:
        print("[DBG] run-mode tracker overlay exception:", e)
        overlay.draw_rois(
            vis,
            rois=roi_mgr_to_list(roi_mgr),
            active_id=roi_mgr.selected_id,
            roi_results=state.last_results,
        )
        state.tracking_stable = False
        state.stable_frame_count = 0