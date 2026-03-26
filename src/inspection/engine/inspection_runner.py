def run_inspection(*, inspector, frame_gray8, auto_mode=False):
    return inspector.inspect(frame_gray8, auto_mode=auto_mode)

def _empty_align_result():
    return {
        "per_roi": {},
        "global": {
            "dx": 0,
            "dy": 0,
            "dangle": 0.0,
            "score": 0.0,
        },
    }