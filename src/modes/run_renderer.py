from ui import overlay_clean as overlay


def _safe_draw_overlay(vis, roi_mgr, state, show_metrics):
    overlay.draw_rois(
        vis,
        rois=roi_mgr_to_list(roi_mgr),
        active_id=roi_mgr.selected_id,
        roi_results=state.last_results,
        compact=True,
        show_metrics=show_metrics,
    )


def _reset_tracking_state(state, stabilizer=None):
    state.tracking_stable = False
    state.stable_frame_count = 0
    state._trk_frame_idx = 0
    state._trk_cache = None
    state._trk_smoothed_cache = None
    state._trk_motion_stable = False
    if stabilizer is not None:
        stabilizer.reset()


def roi_mgr_to_list(roi_mgr):
    return [
        {
            "id": roi.get("id"),
            "label": roi.get("name"),
            "rect": (
                int(roi.get("x", 0)),
                int(roi.get("y", 0)),
                int(roi.get("w", 0)),
                int(roi.get("h", 0)),
            ),
            "angle": float(roi.get("angle", 0.0)),
        }
        for roi in getattr(roi_mgr, "rois", [])
    ]


def draw_roi_overlay(vis, moved, last_results, roi_label_pos, show_metrics=False):
    del roi_label_pos  # kept in the public signature for caller compatibility

    rois = []
    for moved_roi in moved:
        roi_id = str(moved_roi.get("id"))
        rois.append({
            "id": int(moved_roi.get("id")),
            "label": moved_roi.get("name", f"ROI{roi_id}"),
            "rect": (
                int(moved_roi.get("x", 0)),
                int(moved_roi.get("y", 0)),
                int(moved_roi.get("w", 0)),
                int(moved_roi.get("h", 0)),
            ),
            "angle": float(moved_roi.get("angle", 0.0)),
        })

    roi_results = None
    if last_results:
        roi_results = {str(key): value for key, value in last_results.items()}

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
    # Snapshot arguments remain for API compatibility; stable snapshot saving is
    # intentionally disabled in the current delivery build.
    del data_dir, snapshot_cooldown, snapshot_keep, prune_snapshots

    tracker_enabled = bool(runtime_cfg.get("enable_tracker", True))
    module_enabled = bool(product_profile["modules"].get("tracker", True))
    if not tracker_enabled or not module_enabled:
        _reset_tracking_state(state, stabilizer)
        _safe_draw_overlay(vis, roi_mgr, state, show_metrics)
        return

    aligner = getattr(inspector, "aligner", None)

    try:
        if aligner is None or frame_gray8 is None:
            _safe_draw_overlay(vis, roi_mgr, state, show_metrics)
            _reset_tracking_state(state, stabilizer)
            return

        if not hasattr(state, "_trk_frame_idx"):
            _reset_tracking_state(state, stabilizer)

        state._trk_frame_idx += 1
        fresh_estimate = state._trk_cache is None or state._trk_frame_idx % 3 == 0

        if fresh_estimate:
            align_result = aligner.estimate(frame_gray8, roi_mgr)
            state._trk_cache = align_result
        else:
            align_result = state._trk_cache

        source_rois = list(getattr(roi_mgr, "rois", []))
        moved = aligner.apply_to_rois(source_rois, align_result)

        if fresh_estimate:
            smoothed, motion_stable = stabilizer.update(moved)
            state._trk_smoothed_cache = smoothed
            state._trk_motion_stable = bool(motion_stable)

            anchor_ok = any(
                bool(anchor.get("ok"))
                and str(anchor.get("reason", "OK")) == "OK"
                for anchor in (align_result.get("anchors") or [])
            )
            state.tracking_stable = bool(anchor_ok and motion_stable)
            if state.tracking_stable:
                state.stable_frame_count += 1
            else:
                state.stable_frame_count = 0
        else:
            smoothed = state._trk_smoothed_cache or moved

        state.status = (
            "RUN MODE (stable)"
            if state.tracking_stable
            else "RUN MODE (tracking...)"
        )

        draw_roi_overlay(
            vis,
            smoothed,
            state.last_results,
            roi_label_pos,
            show_metrics=show_metrics,
        )

    except Exception as exc:
        print("[DBG] run-mode tracker overlay exception:", exc)
        _safe_draw_overlay(vis, roi_mgr, state, show_metrics)
        _reset_tracking_state(state, stabilizer)
