#!/usr/bin/env python3
import os
import time
import json
from dataclasses import dataclass
from enum import Enum

import cv2
import numpy as np

from capture.camera_gst import CameraGST
from roi.roi_manager import ROIManager
from roi.roi_editor import ROIEditor
from inspection.inspector import Inspector
from inspection.stabilizer import Stabilizer
from inspection.normalize import normalize_frame
from inspection.logger import save_snapshot, save_template_copy

from ui import overlay_clean as overlay  # production overlay
from typing import Optional, Dict, Any
from ui.hud import draw_mode_indicator, draw_dev_hud
from ui.pose_guide import draw_pose_message
from runtime.product_profile_loader import load_product_profile
from data_io.sample_capture import handle_sample_keys, prune_snapshots
from ui.control_bar import render_control_bar, key_to_cmd, button_id_to_cmd
from app.command_executor import execute_command
from inspection.inspect_service import run_inspect_once
from modes.run_renderer import draw_run_tracking

# NOTE: overlay_clean.py 에 draw_text_kr() 있어야 함


# =========================
# Commands
# =========================
class UICmd(Enum):
    NONE = 0
    TOGGLE_MODE = 1
    INSPECT = 2
    AUTOTUNE = 3
    RELOAD = 4
    SAVE = 5
    NEXT = 6
    CLEAR = 7
    QUIT = 8
    DELETE = 9
    TOGGLE_AUTO_INSPECT = 10  # NEW





# =========================
# Config
# =========================
DEV_MODE = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
ROI_DIR      = os.path.join(DATA_DIR, "roi")
ROI_PATH     = os.path.join(ROI_DIR, "roi.json")
RECIPE_PATH  = os.path.join(ROI_DIR, "recipe_static.json")
RUNTIME_CFG_PATH = os.path.join(ROI_DIR, "runtime_config.json")
PRODUCT_PROFILE_PATH = os.path.join(ROI_DIR, "product_profile.json")
LOGS_ROOT    = os.path.join(DATA_DIR, "logs")

DEV    = "/dev/video0"
WIDTH  = 1280
HEIGHT = 720
FPS    = 30

GST_PIPELINE = (
    f"v4l2src device={DEV} ! "
    f"video/x-raw,format=GRAY16_LE,width={WIDTH},height={HEIGHT},framerate={FPS}/1 ! "
    "videoconvert ! video/x-raw,format=GRAY8 ! "
    "appsink drop=true max-buffers=1 sync=false"
)

TEMPLATE_PATH = os.path.join(PROJECT_ROOT, "data", "roi", "align_template.png")

NORMALIZE_ENABLED = False
NORMALIZE_TARGET_MEAN = 120.0

SNAPSHOT_COOLDOWN = 5.0
SNAPSHOT_KEEP = 200

POSE_BAD_N = 5
POSE_ROI_ID_STR = "1"          # pose 판단 ROI (문자열 키)
POSE_METRIC_KEY = "blob_count" # pose 판단 metric
POSE_EXPECT = 4                # blob_count == 4

AUTO_INSPECT_INTERVAL = 0.5    # seconds (2Hz)


# =========================
# Small utils
# =========================
def ensure_dirs():
    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(LOGS_ROOT, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "images", "ok"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "images", "ng"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "templates"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "dataset", "OK"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "dataset", "NG"), exist_ok=True)

def load_runtime_config(path):
    cfg = {
        "enable_auto_inspect": True,
        "auto_inspect_interval": 0.5,
        "auto_inspect_avg5": False,
        "pose_roi_id": "1",
        "pose_metric_key": "blob_count",
        "pose_expect": 4,
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

def roi_label_pos(x, y, w, h, margin=25):
    tx = x
    ty = y - margin
    if ty < 18:
        ty = y + h + 18
    return int(tx), int(ty)

def _extract_info_from_results(results):
    if not results:
        return {}
    total = len(results)
    ng = sum(1 for r in results.values() if not getattr(r, "ok", False))
    info = {"total": total, "ng": ng}
    for r in results.values():
        m = getattr(r, "metrics", None) or {}
        if "norm_gain" in m: info["norm_gain"] = m["norm_gain"]
        if "dx" in m: info["dx"] = m["dx"]
        if "dy" in m: info["dy"] = m["dy"]
        break
    return info


@dataclass
class AppState:
    edit_mode: bool = True
    status: str = "EDIT MODE"
    quit_requested: bool = False
    space_lock: bool = False

    last_results: Optional[Dict[str, Any]] = None   # {"1": Result, ...}
    last_overall_ok: Optional[bool] = None

    pending_cmd: UICmd = UICmd.NONE

    # pose assist
    pose_bad_cnt: int = 0

    # auto inspect
    auto_inspect: bool = False
    last_auto_inspect_ts: float = 0.0

    # UI
    last_buttons: list = None

    # snapshot cooldown
    last_snapshot_time: float = 0.0


class VisionApp:
    def __init__(self):
        ensure_dirs()

        self.runtime_cfg = load_runtime_config(RUNTIME_CFG_PATH)
        self.product_profile = load_product_profile(PRODUCT_PROFILE_PATH)
        self.cam = CameraGST(GST_PIPELINE)
        self.roi_mgr = ROIManager(frame_size=(WIDTH, HEIGHT))
        try:
            self.roi_mgr.load(ROI_PATH)
            print("[DBG ROI COUNT]", len(getattr(self.roi_mgr, "rois", [])))
        except Exception:
            pass

        self.editor = ROIEditor(self.roi_mgr)
        self.inspector = Inspector(self.roi_mgr, recipe_path=RECIPE_PATH, logs_root=LOGS_ROOT)
        self.editor.on_select_changed = self.inspector.reset_tracker_template

        self.stabilizer = Stabilizer(window=5, move_thresh_px=3, alpha=0.7)

        self.state = AppState(last_buttons=[])
        self.state.auto_inspect = bool(self.runtime_cfg.get("enable_auto_inspect", True))

        # load saved tracker template if exists
        self._load_alignment_template()

        self.win = "Static Mode - ROI Setup"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, WIDTH, HEIGHT)
        cv2.setMouseCallback(self.win, self._mouse_router)
        self.UICmd = UICmd

    def _load_alignment_template(self):
        try:
            if os.path.exists(TEMPLATE_PATH):
                tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
                trk = getattr(self.inspector, "tracker", None)
                if tpl is not None and trk is not None and hasattr(trk, "set_template"):
                    trk.set_template(tpl)
                    print("[INFO] alignment template loaded into tracker.")
        except Exception as e:
            print("[WARN] failed to load alignment template:", e)

    # -------------------------
    # Input handlers
    # -------------------------
    def _mouse_router(self, event, x, y, flags, param):
        st = self.state

        if event == cv2.EVENT_MOUSEMOVE:
            if st.edit_mode:
                self.editor._on_mouse(event, x, y, flags, None)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            for b in (st.last_buttons or []):
                x1, y1, x2, y2 = b.get("rect", (0,0,0,0))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if b.get("enabled", True):
                        bid = b.get("id")
                        st.pending_cmd = button_id_to_cmd(bid, UICmd)
                    return

            if st.edit_mode:
                self.editor._on_mouse(event, x, y, flags, None)
            return

        if st.edit_mode:
            self.editor._on_mouse(event, x, y, flags, None)

    # -------------------------
    # Core actions
    # -------------------------
    def _toggle_mode(self):
        st = self.state
        st.edit_mode = not st.edit_mode
        try:
            self.inspector.mean_filter.reset()
        except Exception:
            pass

        # switching into edit: clear runtime template (optional)
        try:
            trk = getattr(self.inspector, "tracker", None)
            if st.edit_mode:
                if trk is not None and hasattr(trk, "set_template"):
                    trk.set_template(None)
            else:
                self._load_alignment_template()
        except Exception:
            pass

        st.status = "EDIT MODE" if st.edit_mode else "RUN MODE"

    def _save_roi_and_template(self, frame_gray8):
        st = self.state
        try:
            self.roi_mgr.save(ROI_PATH)
            try:
                ok_tpl = self.roi_mgr.save_alignment_template(frame_gray8, TEMPLATE_PATH, roi_id=None)
                if ok_tpl:
                    self._load_alignment_template()
                    st.status = "Saved ROI + Template"
                else:
                    st.status = "Saved ROI"
            except Exception:
                st.status = "Saved ROI"
        except Exception as e:
            st.status = f"Save failed: {e}"

    def _toggle_auto_inspect(self):
        st = self.state
        st.auto_inspect = not st.auto_inspect
        st.last_auto_inspect_ts = 0.0
        st.status = "AUTO INSPECT ON" if st.auto_inspect else "AUTO INSPECT OFF"


    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        st = self.state
        self.cam.open()

        last_ok_frame_time = time.time()

        while True:
            frame = self.cam.read()
            if frame is None:
                if time.time() - last_ok_frame_time > 1.0:
                    st.status = "No frames >1s (check camera)"
                cv2.waitKey(1)
                if st.quit_requested:
                    break
                continue

            last_ok_frame_time = time.time()

            if frame.ndim == 3:
                frame = frame[:, :, 0]
            frame_gray8 = frame

            # optional normalize (RUN only)
            if (not st.edit_mode) and NORMALIZE_ENABLED and frame_gray8 is not None:
                frame_gray8, _ = normalize_frame(frame_gray8, target_mean=NORMALIZE_TARGET_MEAN, do_clahe=True)

            vis = cv2.cvtColor(frame_gray8, cv2.COLOR_GRAY2BGR)

            # edit / run draw
            if st.edit_mode:
                self.editor.update(vis)
            else:
                draw_run_tracking(
                    vis,
                    frame_gray8,
                    runtime_cfg=self.runtime_cfg,
                    product_profile=self.product_profile,
                    state=st,
                    roi_mgr=self.roi_mgr,
                    inspector=self.inspector,
                    stabilizer=self.stabilizer,
                    data_dir=DATA_DIR,
                    snapshot_cooldown=SNAPSHOT_COOLDOWN,
                    snapshot_keep=SNAPSHOT_KEEP,
                    prune_snapshots=prune_snapshots,
                    roi_label_pos=roi_label_pos,
                )

                # Auto inspect tick
                cfg = self.runtime_cfg

                if st.auto_inspect and self.product_profile["modules"].get("auto_inspect", True):
                    now = time.time()
                    interval = float(cfg.get("auto_inspect_interval", 0.5))
                    avg5 = bool(cfg.get("auto_inspect_avg5", False))

                    if (now - st.last_auto_inspect_ts) >= interval:
                        st.last_auto_inspect_ts = now
                        run_inspect_once(
                            cam=self.cam,
                            inspector=self.inspector,
                            runtime_cfg=self.runtime_cfg,
                            state=st,
                            frame_gray8=frame_gray8,
                            vis_bgr=vis,
                            avg5=avg5,
                        )

            # status + banner
            overlay.draw_status_bar(vis, st.status)
            if st.last_overall_ok is not None:
                overlay.draw_overall_banner(vis, st.last_overall_ok, info=_extract_info_from_results(st.last_results))

            # control bar (buttons)
            st.last_buttons = render_control_bar(vis, st.edit_mode)

            # HUD
            vis = draw_pose_message(vis, st.pose_bad_cnt, int(self.runtime_cfg.get("pose_bad_n", 5)))
            draw_dev_hud(vis, st, self.product_profile)

            # show
            cv2.imshow(self.win, vis)

            # key
            key = cv2.waitKey(1) & 0xFF

            # sample keys first (consumes key)
            consumed, sample_msg = handle_sample_keys(
                key,
                frame_gray8,
                vis,
                edit_mode=st.edit_mode,
                roi_mgr=self.roi_mgr,
                data_dir=DATA_DIR,
                snapshot_keep=SNAPSHOT_KEEP,
            )
            if consumed:
                if sample_msg:
                    st.status = sample_msg
                key = 255

            # keyboard -> pending cmd
            if key != 255:
                cmd = key_to_cmd(key, UICmd)
                if cmd != UICmd.NONE:
                    st.pending_cmd = cmd

            # execute pending cmd
            if st.pending_cmd != UICmd.NONE:
                cmd_to_run = st.pending_cmd
                st.pending_cmd = UICmd.NONE
                execute_command(self, cmd_to_run, frame_gray8, vis)

            if st.quit_requested:
                break

        self.cam.release()
        cv2.destroyAllWindows()


def main():
    app = VisionApp()
    app.run()

if __name__ == "__main__":
    main()