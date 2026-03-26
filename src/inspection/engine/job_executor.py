import numpy as np

from inspection.inspector import (
    JOB_REGISTRY,
    JOB_RUNNERS,
    JOB_EVALUATORS,
    _run_analyzer_job,
    _job_eval_toolchain,
)
from inspection.score import combined_score


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
):
    job_type = (cfg.get("type") or "").strip().lower()
    orig_margin = None

    if job_type == "washer_presence":
        orig_margin = getattr(inspector.tracker, "search_margin", None)
        inspector.tracker.search_margin = int(cfg.get("tracker_margin", 50))

    registry_pair = JOB_REGISTRY.get(job_type)
    if registry_pair is not None:
        runner, evaluator = registry_pair
    else:
        runner = JOB_RUNNERS.get(job_type, _run_analyzer_job)
        evaluator = JOB_EVALUATORS.get(job_type, _job_eval_toolchain)

    ok, metrics, reason = runner(crop, cfg)

    if metrics is None:
        metrics = {}

    mean_raw = float(np.mean(crop))
    metrics["mean_raw"] = mean_raw
    metrics["mean"] = mean_filter.update(mean_raw)

    need_score = str(cfg.get("type", "")).lower() in ("mean_score", "score_threshold", "texture_score")
    if need_score:
        try:
            score = combined_score(crop)
        except Exception:
            score = 0.0
        metrics["score"] = float(score)

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