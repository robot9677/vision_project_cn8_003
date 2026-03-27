import numpy as np
try:
    from inspection.score import combined_score
except:
    combined_score = None


def execute_inspection_job(
    *,
    inspector,
    crop,
    cfg,
    recipe_default,
    runtime_cfg,
    mean_filter,
    norm_gain,
    roi_dx,
    roi_dy,
    roi_dangle,
    pose,
    trk_score,
    job_registry,
    job_runners,
    job_evaluators,
    default_runner,
    default_evaluator,
):
    job_type = (cfg.get("type") or "").strip().lower()
    orig_margin = None

    if job_type == "washer_presence":
        orig_margin = getattr(inspector.tracker, "search_margin", None)
        inspector.tracker.search_margin = int(cfg.get("tracker_margin", 50))

    registry_pair = job_registry.get(job_type)
    if registry_pair is not None:
        runner, evaluator = registry_pair
    else:
        runner = job_runners.get(job_type, default_runner)
        evaluator = job_evaluators.get(job_type, default_evaluator)

    ok, metrics, reason = runner(crop, cfg)

    if metrics is None:
        metrics = {}

    mean_raw = float(np.mean(crop))
    metrics["mean_raw"] = mean_raw
    metrics["mean"] = mean_filter.update(mean_raw)

    need_score = str(cfg.get("type", "")).lower() in ("mean_score", "score_threshold", "texture_score")
    if need_score:
        if combined_score is not None:
            try:
                score = combined_score(crop)
            except Exception:
                score = 0.0
        else:
            score = 0.0

    metrics["norm_gain"] = float(norm_gain)
    metrics["dx"] = roi_dx
    metrics["dy"] = roi_dy
    metrics["dangle"] = float(roi_dangle)
    metrics["trk_score"] = float(pose.get("score", trk_score))
    metrics["align_anchor_id"] = pose.get("anchor_id")
    metrics["inspection_id"] = cfg.get("id", "job")

    job_ok, job_reason = evaluator(
        ok=ok,
        metrics=metrics,
        reason=reason,
        cfg=cfg,
        recipe_default=recipe_default,
        runtime_cfg=runtime_cfg,
    )

    if orig_margin is not None:
        inspector.tracker.search_margin = orig_margin

    return job_ok, metrics, job_reason, job_type