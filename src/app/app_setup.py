import os


def ensure_dirs(data_dir, roi_dir, logs_root):
    os.makedirs(roi_dir, exist_ok=True)
    os.makedirs(logs_root, exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images", "ok"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "images", "ng"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "templates"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dataset", "OK"), exist_ok=True)
    os.makedirs(os.path.join(data_dir, "dataset", "NG"), exist_ok=True)