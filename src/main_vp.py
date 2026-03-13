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

from ui import overlay_clean as overlay
from typing import Optional, Dict, Any
from ui.hud import draw_mode_indicator, draw_dev_hud
from ui.pose_guide import draw_pose_message
from runtime.product_profile_loader import load_product_profile
from data_io.sample_capture import handle_sample_keys, prune_snapshots
from ui.control_bar import render_control_bar, key_to_cmd, button_id_to_cmd
from app.command_executor import execute_command
from inspection.inspect_service import run_inspect_once
from modes.run_renderer import draw_run_tracking
from runtime.runtime_config_loader import load_runtime_config
from app.app_setup import ensure_dirs
from app.app_paths import (
    PRODUCT_PROFILE_PATH,
    TEMPLATE_PATH,
    DATA_DIR,
    ROI_DIR,
    ROI_PATH,
    RECIPE_PATH,
    RUNTIME_CONFIG_PATH,
    LOGS_ROOT,
    PROJECT_ROOT,
)

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

POSE_ROI_ID_STR = "1"          # pose 판단 ROI (문자열 키)
POSE_METRIC_KEY = "blob_count" # pose 판단 metric
POSE_EXPECT = 4                # blob_count == 4

# =========================
# Small utils
# =========================

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

    tracking_stable: bool = False
    stable_frame_count: int = 0
    run_mode_text: str = "HELD"


class VisionApp:
    def __init__(self):
        ensure_dirs(DATA_DIR, ROI_DIR, LOGS_ROOT)

        self.runtime_cfg = load_runtime_config(RUNTIME_CONFIG_PATH)
        self.product_profile = load_product_profile(PRODUCT_PROFILE_PATH)
        self.cam = CameraGST(GST_PIPELINE,auto_brightness=False,denoise_method='none',)
        self.roi_mgr = ROIManager(frame_size=(WIDTH, HEIGHT))
        try:
            self.roi_mgr.load(ROI_PATH)
            if DEV_MODE == True:
                print("[DBG ROI COUNT]", len(getattr(self.roi_mgr, "rois", [])))
        except Exception as e:
            pass

        self.editor = ROIEditor(self.roi_mgr)
        self.inspector = Inspector(
            self.roi_mgr,
            recipe_path=RECIPE_PATH,
            logs_root=LOGS_ROOT,
            runtime_cfg=self.runtime_cfg,
        )
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
        except Exception as e:
            pass

        # switching into edit: clear runtime template (optional)
        try:
            trk = getattr(self.inspector, "tracker", None)
            if st.edit_mode:
                if trk is not None and hasattr(trk, "set_template"):
                    trk.set_template(None)
            else:
                self._load_alignment_template()
        except Exception as e:
            pass

        if st.edit_mode:
            st.status = "EDIT MODE"
            self.last_overall_ok = None
            self.last_results = None
        else:
            run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()
            st.status = "RUN MODE / STATIC" if run_mode == "static" else "RUN MODE / HELD"
            st.run_mode_text = "STATIC" if run_mode == "static" else "HELD"

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
            except Exception as e:
                st.status = "Saved ROI"
        except Exception as e:
            st.status = f"Save failed: {e}"

    def _toggle_auto_inspect(self):
        st = self.state
        st.auto_inspect = not st.auto_inspect
        st.last_auto_inspect_ts = 0.0
        run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()
        mode_text = "STATIC" if run_mode == "static" else "HELD"
        st.status = f"AUTO INSPECT ON / {mode_text}" if st.auto_inspect else f"AUTO INSPECT OFF / {mode_text}"

    def _run_auto_inspect_tick(self, frame_gray8, vis_bgr):
        st = self.state
        cfg = self.runtime_cfg

        if not (st.auto_inspect and self.product_profile["modules"].get("auto_inspect", True)):
            return

        now = time.time()
        interval = float(cfg.get("auto_inspect_interval", 0.5))
        avg5 = bool(cfg.get("auto_inspect_avg5", False))
        stable_required = int(cfg.get("auto_inspect_stable_frames", 3))
        run_mode = str(cfg.get("run_mode", "held")).lower()

        allow_inspect = False
        if run_mode == "static":
            allow_inspect = True
        else:
            allow_inspect = st.tracking_stable and st.stable_frame_count >= stable_required

        if allow_inspect and (now - st.last_auto_inspect_ts) >= interval:
            st.last_auto_inspect_ts = now
            run_inspect_once(
                cam=self.cam,
                inspector=self.inspector,
                runtime_cfg=self.runtime_cfg,
                state=st,
                frame_gray8=frame_gray8,
                vis_bgr=vis_bgr,
                avg5=avg5,
            )

    def _render_run_frame(self, vis, frame_gray8):
        st = self.state
        run_mode = str(self.runtime_cfg.get("run_mode", "held")).lower()

        if run_mode == "static":

            overlay.draw_rois(
                vis,
                rois=[
                    {
                        "id": r.get("id"),
                        "label": r.get("name"),
                        "rect": (
                            int(r.get("x", 0)),
                            int(r.get("y", 0)),
                            int(r.get("w", 0)),
                            int(r.get("h", 0)),
                        ),
                        "angle": float(r.get("angle", 0.0)),
                    }
                    for r in getattr(self.roi_mgr, "rois", [])
                ],
                active_id=self.roi_mgr.selected_id,
                roi_results=st.last_results,
            )

            st.tracking_stable = True
            st.stable_frame_count = 999

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
                snapshot_cooldown=float(self.runtime_cfg.get("snapshot_cooldown", 5.0)),
                snapshot_keep=int(self.runtime_cfg.get("snapshot_keep", 200)),
                prune_snapshots=prune_snapshots,
                roi_label_pos=roi_label_pos,
            )

    def _handle_key_input(self, key, frame_gray8, vis_bgr):
        st = self.state

        consumed, sample_msg = handle_sample_keys(
            key,
            frame_gray8,
            vis_bgr,
            edit_mode=st.edit_mode,
            roi_mgr=self.roi_mgr,
            data_dir=DATA_DIR,
            snapshot_keep=int(self.runtime_cfg.get("snapshot_keep", 200)),
        )
        if consumed:
            if sample_msg:
                st.status = sample_msg
            return

        if key != -1:
            cmd = key_to_cmd(key, UICmd)
            if cmd != UICmd.NONE:
                st.pending_cmd = cmd

        if st.pending_cmd != UICmd.NONE:
            cmd_to_run = st.pending_cmd
            st.pending_cmd = UICmd.NONE
            execute_command(self, cmd_to_run, frame_gray8, vis_bgr)

    def _draw_ui(self, vis):
        st = self.state
        st.active_roi_id = self.roi_mgr.selected_id

        overlay.draw_status_bar(vis, st.status)

        if (not st.edit_mode) and (st.last_overall_ok is not None):
            overlay.draw_overall_banner(vis,st.last_overall_ok,info=_extract_info_from_results(st.last_results),)

        st.last_buttons = render_control_bar(vis, st.edit_mode)

        if bool(self.runtime_cfg.get("enable_pose_guide", True)):
            vis = draw_pose_message(
                vis,
                st.pose_bad_cnt,
                int(self.runtime_cfg.get("pose_bad_n", 5)),
            )

        draw_mode_indicator(vis, st.edit_mode)
        draw_dev_hud(vis, st, self.product_profile)
        return vis
    
    def _prepare_frame(self, frame):
        st = self.state

        if frame is None:
            return None, None

        if frame.ndim == 3:
            frame = frame[:, :, 0]

        frame_gray8 = frame

        if (not st.edit_mode) and frame_gray8 is not None:
            norm_enabled = bool(self.runtime_cfg.get("normalize_enabled", False))
            norm_target = float(self.runtime_cfg.get("normalize_target_mean", 120.0))
            if norm_enabled:
                frame_gray8, _ = normalize_frame(
                    frame_gray8,
                    target_mean=norm_target,
                    do_clahe=True,
                )

        vis = cv2.cvtColor(frame_gray8, cv2.COLOR_GRAY2BGR)
        return frame_gray8, vis
    
    def _read_frame(self):
        frame = self.cam.read()
        if frame is None:
            return None
        return frame

    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        st = self.state
        self.cam.open()

        last_ok_frame_time = time.time()

        while True:
            frame = self._read_frame()
            if frame is None:
                if time.time() - last_ok_frame_time > 1.0:
                    st.status = "No frames >1s (check camera)"
                cv2.waitKey(1)
                if st.quit_requested:
                    break
                continue

            last_ok_frame_time = time.time()

            frame_gray8, vis = self._prepare_frame(frame)
            if frame_gray8 is None:
                cv2.waitKey(1)
                if st.quit_requested:
                    break
                continue

            # edit / run draw
            if st.edit_mode:
                self.editor.update(vis)
            else:
                self._render_run_frame(vis, frame_gray8)

            if not st.edit_mode:
                self._run_auto_inspect_tick(frame_gray8, vis)
        
            # Status + banner + control Bar(button) + HUD  Draw
            vis = self._draw_ui(vis)

            # show
            cv2.imshow(self.win, vis)

            # key
            key = cv2.waitKey(1) & 0xFF
            self._handle_key_input(key, frame_gray8, vis)

            if st.quit_requested:
                break

        self.cam.release()
        cv2.destroyAllWindows()


def main():
    app = VisionApp()
    app.run()

if __name__ == "__main__":
    main()