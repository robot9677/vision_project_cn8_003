import os
import json


def load_runtime_config(path):
    cfg = {
        "enable_auto_inspect": True,
        "auto_inspect_interval": 0.5,
        "auto_inspect_avg5": False,
        "auto_inspect_stable_frames": 3,
        "run_mode": "held",
        "pose_bad_n": 5,
        "pose_roi_id": "1",
        "pose_metric_key": "blob_count",
        "pose_expect": 4,
        "enable_pose_guide": True,
        "enable_tracker": True,
        "normalize_enabled": False,
        "normalize_target_mean": 120.0,
        "snapshot_cooldown": 5.0,
        "snapshot_keep": 200,
        "tracker_search_margin": 80,
        "tracker_thr": 0.70,
        "tracker_reacquire_margin": 220,
        "tracker_reacquire_scale": 0.5,
        "autotune_target_mean": 50.0,
        "autotune_margin": 10.0,
    }

    try:
        if os.path.exists(path):
            with open(path, "r", encoding="utf-8") as f:
                user_cfg = json.load(f)
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
    except Exception as e:
        print("[WARN] runtime config load failed:", e)

    return cfg