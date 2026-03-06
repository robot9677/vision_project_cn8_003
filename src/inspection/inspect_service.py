import numpy as np


def run_inspect_once(
    *,
    cam,
    inspector,
    runtime_cfg,
    state,
    frame_gray8,
    vis_bgr,
    avg5=True,
):
    pose_roi_id = str(runtime_cfg.get("pose_roi_id", "1"))
    pose_metric_key = str(runtime_cfg.get("pose_metric_key", "blob_count"))
    pose_expect = int(runtime_cfg.get("pose_expect", 4))

    if avg5:
        frames = []
        for _ in range(5):
            f = cam.read()
            if f is None:
                continue
            if getattr(f, "ndim", 0) == 3:
                f = f[:, :, 0]
            frames.append(f)
        avg = np.mean(frames, axis=0).astype("uint8") if frames else frame_gray8
    else:
        avg = frame_gray8

    overall_ok, results = inspector.inspect(avg, auto_mode=state.auto_inspect)

    try:
        inspector.save_run(avg, vis_bgr.copy(), overall_ok, results)
    except Exception as e:
        print("[DBG] save_run failed:", e)

    state.last_results = {str(k): v for k, v in results.items()} if results else {}
    state.last_overall_ok = overall_ok
    state.status = f"INSPECT {'OK' if overall_ok else 'NG'}"

    try:
        inspector.log_result(state.last_overall_ok, state.last_results)
    except Exception:
        pass

    r = (state.last_results or {}).get(pose_roi_id)
    bc = None
    if r is not None and hasattr(r, "metrics"):
        bc = (r.metrics or {}).get(pose_metric_key, None)
    elif isinstance(r, dict):
        bc = (r.get("metrics") or {}).get(pose_metric_key, None)

    if bc is None:
        state.pose_bad_cnt = 0
    else:
        if int(bc) == pose_expect:
            state.pose_bad_cnt = 0
        else:
            state.pose_bad_cnt += 1

    return overall_ok, state.last_results