def run_qr(crop, cfg):
    from inspection.inspector import _run_qr_job
    return _run_qr_job(crop, cfg)

def eval_qr(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    from inspection.inspector import _job_eval_qr_presence
    return _job_eval_qr_presence(ok, metrics, reason, cfg, recipe_default, runtime_cfg)