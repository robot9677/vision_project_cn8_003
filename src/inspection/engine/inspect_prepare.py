from .inspection_runner import _empty_align_result
from inspection.preprocess import normalize_by_roi


def prepare_inspection_context(*, inspector, frame_gray8, auto_mode=False):
    ref = inspector.roi_mgr.get(1)
    norm_gain = 1.0
    trk_score = 0.0
    align_result = None

    if ref is not None:
        ref_id = ref["id"]
        ref_crop_raw = inspector.roi_mgr.crop(frame_gray8, ref_id)

        use_normalize = bool(inspector.runtime_cfg.get("normalize_enabled", False))
        if use_normalize and ref_crop_raw is not None and ref_crop_raw.size > 0:
            target_mean = float(inspector.runtime_cfg.get("normalize_target_mean", 50.0))
            frame_gray8, norm_gain = normalize_by_roi(frame_gray8, ref_crop_raw, target_mean=target_mean)
        else:
            norm_gain = 1.0

    use_tracker = bool(inspector.runtime_cfg.get("enable_tracker", True))

    if use_tracker and getattr(inspector, "aligner", None) is not None:
        align_result = inspector.aligner.estimate(frame_gray8, inspector.roi_mgr)
        g = align_result.get("global") or {}
        trk_score = float(g.get("score", 0.0))

        if not auto_mode:
            anchors = align_result.get("anchors") or []
            if anchors:
                dbg = " ".join(
                    f"{a.get('id')}[ok={a.get('ok')} dx={a.get('dx')} dy={a.get('dy')} da={a.get('dangle'):.2f} sc={a.get('score'):.3f}]"
                    for a in anchors
                )
                print(f"[DBG ALIGN] {dbg}")
    else:
        align_result = _empty_align_result()

    return {
        "frame_gray8": frame_gray8,
        "norm_gain": norm_gain,
        "trk_score": trk_score,
        "align_result": align_result,
    }