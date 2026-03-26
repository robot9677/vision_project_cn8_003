def run_ocr(crop, cfg):
    from inspection.inspector import _run_ocr_text
    return _run_ocr_text(crop, cfg)

def eval_ocr(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    from inspection.inspector import _job_eval_ocr_text
    return _job_eval_ocr_text(ok, metrics, reason, cfg, recipe_default, runtime_cfg)