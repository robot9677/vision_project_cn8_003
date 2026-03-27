import numpy as np
from inspection.engine.inspection_runner import run_inspection


def _capture_avg_frame(cam, fallback_frame, avg5=True):
    if not avg5:
        return fallback_frame

    frames = []
    for _ in range(5):
        f = cam.read()
        if f is None:
            continue
        if getattr(f, "ndim", 0) == 3:
            f = f[:, :, 0]
        frames.append(f)

    return np.mean(frames, axis=0).astype("uint8") if frames else fallback_frame


def _store_inspect_result(state, overall_ok, results):
    state.last_results = {str(k): v for k, v in results.items()} if results else {}
    state.last_overall_ok = overall_ok
    state.status = f"INSPECT {'OK' if overall_ok else 'NG'}"


def _update_pose_bad_count(state, runtime_cfg):
    pose_roi_id = str(runtime_cfg.get("pose_roi_id", "1"))
    pose_metric_key = str(runtime_cfg.get("pose_metric_key", "blob_count"))
    pose_expect = int(runtime_cfg.get("pose_expect", 4))

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
    avg = _capture_avg_frame(cam, frame_gray8, avg5=avg5)

    # 최초 초기화
    if not hasattr(state, "_inspect_frame_idx"):
        state._inspect_frame_idx = 0
        state._inspect_cache = None

    state._inspect_frame_idx += 1

    # 3프레임마다만 inspect 실행
    if state._inspect_frame_idx % 3 == 0 or state._inspect_cache is None:
        overall_ok, results = run_inspection(
            inspector=inspector,
            frame_gray8=avg,
            auto_mode=state.auto_inspect,
        )
        state._inspect_cache = (overall_ok, results)
    else:
        overall_ok, results = state._inspect_cache

    try:
        inspector.save_run(avg, vis_bgr.copy(), overall_ok, results)
    except Exception as e:
        print("[DBG] save_run failed:", e)

    _store_inspect_result(state, overall_ok, results)

    try:
        inspector.log_result(state.last_overall_ok, state.last_results)
    except Exception:
        pass

    _update_pose_bad_count(state, runtime_cfg)

    return overall_ok, state.last_results