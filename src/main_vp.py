#!/usr/bin/env python3
import os
import time
import json
from enum import Enum
import sys
import subprocess
from dataclasses import dataclass

import cv2
import numpy as np

from capture.camera_factory import create_camera_from_hardware_config
from hardware.hardware_config_loader import load_hardware_config
from roi.roi_manager import ROIManager
from roi.roi_editor import ROIEditor
from inspection.inspector import Inspector
from inspection.stabilizer import Stabilizer
from inspection.normalize import normalize_frame
from inspection.logger import save_snapshot, save_template_copy

from plc.plc_config_loader import load_plc_config
from plc.plc_controller import create_plc_controller
from plc.plc_error_logger import save_plc_error_log, save_plc_test_log

from light.light_controller import create_light_controller_from_hardware_config

from ui import overlay_clean as overlay
from typing import Optional, Dict, Any
from ui.hud import draw_mode_indicator, draw_dev_hud
from ui.pose_guide import draw_pose_message
from ui.service_panel import ServicePanel
from runtime.product_profile_loader import load_product_profile
from data_io.sample_capture import handle_sample_keys, prune_snapshots
from ui.control_bar import render_control_bar, key_to_cmd, button_id_to_cmd
from app.command_executor import execute_command
from inspection.inspect_service import run_inspect_once
from modes.run_renderer import draw_run_tracking
from runtime.runtime_config_loader import load_runtime_config
from app.app_setup import ensure_dirs
from app.app_paths import (
    PLC_CONFIG_PATH,
    HARDWARE_CONFIG_PATH,
    PRODUCT_PROFILE_PATH,
    PROFILES_DIR,
    TEMPLATE_PATH,
    DATA_DIR,
    ROI_DIR,
    ROI_PATH,
    RECIPE_PATH,
    RECIPES_DIR,
    DEFAULT_RECIPE_PATH,
    RUNTIME_CONFIG_PATH,
    LOGS_ROOT,
    PROJECT_ROOT,
)
from inspection.auto_baseline import AutoBaseline

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
    TOGGLE_SERVICE_PANEL = 11

# =========================
# Config
# =========================
DEV_MODE = True

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



class LightCommunicationError(RuntimeError):
    pass
class InspectionResultError(RuntimeError):
    pass
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

    # spot light pre-arm
    spot_armed: bool = False
    spot_armed_ts: float = 0.0
    spot_armed_brightness: int = 0

    tracking_stable: bool = False
    stable_frame_count: int = 0
    run_mode_text: str = "HELD"

    #학습을 위한 저장
    baseline_learning: bool = False
    baseline_target_count: int = 10
    baseline_count: int = 0

    # PLC shutdown
    plc_shutdown_requested: bool = False
    plc_shutdown_started: bool = False

    # Vision error latch
    camera_error_latched: bool = False
    light_error_latched: bool = False

    camera_error_grace_until: float = 0.0

    # PLC forced-error test state
    test_error_active: bool = False
    test_error_type: str = ""
    test_error_code: int = 0
    test_error_injected_at: str = ""
    test_error_request_id: str = ""
    test_error_safe_logical: bool = True
    test_error_phase: str = "IDLE"
    test_error_result: str = ""
    test_error_log_path: str = ""
    test_error_log_saved: bool = False
    test_error_recovered_at: str = ""
    latest_error_log_path: str = ""


class VisionApp:
    def __init__(self):
        ensure_dirs(DATA_DIR, ROI_DIR, LOGS_ROOT)

        self.runtime_cfg = load_runtime_config(RUNTIME_CONFIG_PATH)
        self.hardware_cfg = load_hardware_config(HARDWARE_CONFIG_PATH)
        self.plc_cfg = load_plc_config(PLC_CONFIG_PATH)

        profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()
        profile_path = PRODUCT_PROFILE_PATH

        if profile_name:
            candidate = os.path.join(PROFILES_DIR, f"{profile_name}_profile.json")
            if os.path.exists(candidate):
                profile_path = candidate

        self.product_profile = load_product_profile(profile_path)
        print("[PROFILE]", profile_path)

        self.cam, self.camera_info = create_camera_from_hardware_config(self.hardware_cfg)

        self.frame_width = int(self.camera_info.get("width", 1280))
        self.frame_height = int(self.camera_info.get("height", 720))

        camera_profile = self.product_profile.get(
            "camera_profile",
            self.camera_info.get("camera_profile", "default"),
        )

        if hasattr(self.cam, "set_profile"):
            self.cam.set_profile(camera_profile)

        print("[MAIN] camera created", self.camera_info.get("name", "camera"))
        print("[MAIN] camera device", self.camera_info.get("device", ""))
        print("[MAIN] camera size", self.frame_width, self.frame_height)

        self.runtime_cfg["_product_profile"] = self.product_profile
        self.runtime_cfg["_hardware_config"] = self.hardware_cfg
        self.runtime_cfg["_camera_info"] = self.camera_info

        self.plc = create_plc_controller(self.plc_cfg)
        self.runtime_cfg["_plc_config"] = self.plc_cfg

        self.light = create_light_controller_from_hardware_config(self.hardware_cfg)
        self.runtime_cfg["_light_state"] = self.light.get_state()
        
        recipe_name = self.product_profile.get("recipe_name", "tape_presence")
        recipe_candidate = os.path.join(RECIPES_DIR, f"{recipe_name}.json")
        selected_recipe_path = recipe_candidate if os.path.exists(recipe_candidate) else DEFAULT_RECIPE_PATH
        self.roi_mgr = ROIManager(frame_size=(self.frame_width, self.frame_height))

        profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()

        roi_path = ROI_PATH  # fallback

        if profile_name:
            candidate = os.path.join(PROFILES_DIR, f"{profile_name}_roi.json")
            if os.path.exists(candidate):
                roi_path = candidate

        print("[ROI PATH]", roi_path)

        try:
            self.roi_mgr.load(roi_path)
            if DEV_MODE == True:
                print("[DBG ROI COUNT]", len(getattr(self.roi_mgr, "rois", [])))
        except Exception as e:
            pass

        self.editor = ROIEditor(self.roi_mgr)
        self.inspector = Inspector(
            self.roi_mgr,
            recipe_path=selected_recipe_path,
            logs_root=LOGS_ROOT,
            runtime_cfg=self.runtime_cfg,
        )
        self.editor.on_select_changed = self.inspector.reset_tracker_template

        stab_cfg = self.runtime_cfg.get("stabilizer", {})

        self.stabilizer = Stabilizer(
            window=int(stab_cfg.get("window", 5)),
            move_thresh_px=float(stab_cfg.get("move_thresh_px", 3)),
            alpha=float(stab_cfg.get("alpha", 0.7)),
        )

        self.state = AppState(last_buttons=[])
        self.state.auto_inspect = bool(self.runtime_cfg.get("enable_auto_inspect", True))
        self.service_panel = ServicePanel(self.plc_cfg.get("service_panel", {}) or {})

        # load saved tracker template if exists
        self._load_alignment_template()

        self.win = "Static Mode - ROI Setup"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, self.frame_width, self.frame_height)
        cv2.setMouseCallback(self.win, self._mouse_router)
        self.UICmd = UICmd
        self.baseline_path = os.path.join(ROI_DIR, "baseline_profile.json")
        self.baseline = AutoBaseline(self.baseline_path)
        self.baseline_debug = False
        self._inspect_frame_idx = 0
        self._plc_test_last_request_id = ""

    def _get_primary_anchor_roi_id(self):
        align_cfg = self.product_profile.get("align", {}) or {}
        anchors = align_cfg.get("anchors") or []
        for a in anchors:
            if isinstance(a, dict) and bool(a.get("enabled", True)):
                roi_id = a.get("roi_id")
                if roi_id is not None:
                    return int(roi_id)
        return self.roi_mgr.selected_id
    
    def _update_baseline_from_ok_results(self):
        st = self.state

        if not st.last_overall_ok:
            print("[BASELINE UPDATE] skipped: last result is not OK")
            return False

        if not st.last_results:
            print("[BASELINE UPDATE] skipped: no last_results")
            return False

        updater = AutoBaseline(self.baseline_path)
        if not updater.load():
            print(f"[BASELINE UPDATE] skipped: baseline file not found -> {self.baseline_path}")
            return False

        updated_count = 0
        max_count = int(self.runtime_cfg.get("baseline_max_count", 200))

        for roi_id, res in st.last_results.items():
            metrics = getattr(res, "metrics", None) or {}
            roi_name = f"ROI{roi_id}"

            if roi_name in ("ROI2", "ROI3", "ROI4", "ROI5"):
                v = metrics.get("dark_ratio", None)
                if v is not None:
                    updater.update_from_ok_result(
                        roi_name, "dark_ratio", v, max_count=max_count
                    )
                    updated_count += 1

            elif roi_name == "ROI6":
                v = metrics.get("blob_count", None)
                if v is None:
                    v = metrics.get("blob", None)
                if v is not None:
                    updater.update_from_ok_result(
                        roi_name, "blob_count", v, max_count=max_count
                    )
                    updated_count += 1

        if updated_count <= 0:
            print("[BASELINE UPDATE] skipped: no valid metrics extracted")
            return False

        if not updater.stats:
            print("[BASELINE UPDATE] skipped: updater.stats empty")
            return False

        updater.save()
        self.baseline = updater
        print(f"[BASELINE UPDATE] updated features = {updated_count}")
        return True
    
    def _load_alignment_template(self):
        try:
            aligner = getattr(self.inspector, "aligner", None)
            loaded = 0
            if aligner is not None and hasattr(aligner, "load_templates_from_disk"):
                loaded = int(aligner.load_templates_from_disk() or 0)
            if loaded > 0:
                print(f"[INFO] alignment template loaded: {loaded}")
            elif os.path.exists(TEMPLATE_PATH):
                tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
        except Exception as e:
            print("[WARN] failed to load alignment template:", e)

    # -------------------------
    # Input handlers
    # -------------------------
    def _mouse_router(self, event, x, y, flags, param):
        st = self.state

        if self.service_panel.handle_mouse(event, x, y):
            action = self.service_panel.pop_action()
            if action:
                self._handle_service_action(action)
            return

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
                        if bid == "quit":
                            st.quit_requested = True
                        else:
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
                if getattr(self.inspector, "aligner", None) is not None:
                    self.inspector.aligner.reset_templates()
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
            profile_name = str(self.runtime_cfg.get("profile_name", "") or "").strip()
            recipe_name = str(self.product_profile.get("recipe_name", "") or "").strip()

            roi_path = ROI_PATH
            if profile_name:
                roi_path = os.path.join(PROFILES_DIR, f"{profile_name}_roi.json")

            tpl_path = TEMPLATE_PATH
            if recipe_name:
                tpl_path = os.path.join(PROFILES_DIR, f"align_template_{recipe_name}.png")

            self.roi_mgr.save(roi_path)

            anchor_roi_id = self._get_primary_anchor_roi_id()
            ok_tpl = self.roi_mgr.save_alignment_template(
                frame_gray8, tpl_path, roi_id=anchor_roi_id
            )
            if ok_tpl:
                if getattr(self.inspector, "aligner", None) is not None:
                    self.inspector.aligner.reset_templates()
                self._load_alignment_template()
                st.status = f"Saved ROI + Align Template (ROI{anchor_roi_id})"
            else:
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

    def _toggle_service_panel(self):
        self.service_panel.request_toggle()

    def _handle_service_action(self, action: str):
        action = str(action or "").strip().lower()
        if action.startswith("inject:"):
            error_type = action.split(":", 1)[1].strip()
            request_id = f"ui-{int(time.time() * 1000)}"
            ok = self._inject_plc_test_error(
                error_type=error_type,
                request_id=request_id,
                custom_message="Service panel logical forced-error test",
            )
            if ok:
                self.service_panel.set_message(
                    f"{error_type.upper()} ERROR INJECTED - WAIT D200=3",
                    3.0,
                )
            else:
                self.service_panel.set_message(
                    "TEST REJECTED - CLEAR CURRENT ERROR FIRST",
                    3.0,
                )

    def _get_service_test_summary(self) -> Dict[str, Any]:
        st = self.state
        test_cfg = self._get_plc_error_test_cfg()
        return {
            "enabled": bool(test_cfg.get("enabled", False)),
            "phase": str(st.test_error_phase),
            "type": str(st.test_error_type),
            "result": str(st.test_error_result),
            "log_saved": bool(st.test_error_log_saved),
            "request_id": str(st.test_error_request_id),
            "injected_at": str(st.test_error_injected_at),
            "recovered_at": str(st.test_error_recovered_at),
        }

    def _save_plc_test_event(
        self,
        phase: str,
        result: str,
        message: str,
        extra: Optional[Dict[str, Any]] = None,
    ) -> Optional[str]:
        st = self.state
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {"snapshot_failed": True, "error": str(e)}

        path = save_plc_test_log(
            logs_root=LOGS_ROOT,
            test_id=str(st.test_error_request_id),
            test_type=str(st.test_error_type),
            phase=str(phase),
            result=str(result),
            error_code=int(st.test_error_code),
            message=str(message),
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            extra=extra or {},
        )
        if path:
            st.test_error_log_path = str(path)
            st.test_error_log_saved = True
        return path

    def _run_auto_inspect_tick(self, frame_gray8, vis_bgr):
        st = self.state
        cfg = self.runtime_cfg

        # spot 조명 검사 모드에서는 auto inspect 금지
        # 수동/PLC pre-arm 테스트 중 의도치 않은 추가 검사 방지
        try:
            spot_cfg = self._get_spot_light_cfg()
            if bool(spot_cfg.get("enabled", False)):
                return
        except Exception:
            pass

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
                use_cache=True,
                cache_every_n=int(cfg.get("auto_inspect_every_n", 3)),
            )

    def _get_spot_light_cfg(self):
        light_sets = self.hardware_cfg.get("light_sets", {}) or {}
        active_light_set = str(self.hardware_cfg.get("active_light_set", "") or "").strip()
        light_cfg = light_sets.get(active_light_set, {}) or {}
        spot_cfg = light_cfg.get("spot_inspect", {}) or {}

        return {
            "enabled": bool(spot_cfg.get("enabled", False)),
            "light_id": str(spot_cfg.get("light_id", "light1")),

            "idle_brightness": int(spot_cfg.get("idle_brightness", 20)),
            "inspect_brightness": int(spot_cfg.get("inspect_brightness", 90)),

            "ramp_enabled": bool(spot_cfg.get("ramp_enabled", True)),
            "ramp_steps": list(spot_cfg.get("ramp_steps", [20, 40, 60, 80, 90])),
            "ramp_step_ms": int(spot_cfg.get("ramp_step_ms", 200)),

            "pre_ready_settle_ms": int(spot_cfg.get("pre_ready_settle_ms", 400)),
            "pre_ready_flush_frames": int(spot_cfg.get("pre_ready_flush_frames", 8)),

            "inspect_settle_ms": int(spot_cfg.get("inspect_settle_ms", 50)),
            "inspect_flush_frames": int(spot_cfg.get("inspect_flush_frames", 2)),

            "armed_timeout_ms": int(spot_cfg.get("armed_timeout_ms", 3000)),
            "restore_after_inspect": bool(spot_cfg.get("restore_after_inspect", True)),
        }

    def _flush_camera_frames(self, n):
        last_gray = None
        last_vis = None

        for _ in range(max(0, int(n))):
            frame = self._read_frame()
            if frame is None:
                continue

            g, v = self._prepare_frame(frame)
            if g is not None and v is not None:
                last_gray = g
                last_vis = v

        return last_gray, last_vis

    def _set_light_brightness_checked(
        self,
        light_id: str,
        brightness: int,
        action: str,
    ):
        ok = self.light.set_brightness(
            light_id,
            brightness,
        )

        light_state = self.light.get_state() or {}
        self.runtime_cfg["_light_state"] = light_state

        info = light_state.get(light_id, {})
        state_ok = (
            info.get("ok")
            if isinstance(info, dict)
            else None
        )
        reason = (
            info.get("reason", "")
            if isinstance(info, dict)
            else ""
        )

        if ok is not True or state_ok is False:
            raise LightCommunicationError(
                f"{action} failed: "
                f"light_id={light_id}, "
                f"brightness={brightness}, "
                f"reason={reason or 'UNKNOWN'}"
            )

        return True

    def _spot_prearm(self, trigger="PLC"):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            return False

        light_id = cfg["light_id"]
        inspect_brightness = int(cfg["inspect_brightness"])

        print(
            f"[LIGHT PREARM] trigger={trigger} "
            f"idle={cfg['idle_brightness']}% inspect={inspect_brightness}% "
            f"timeout={cfg['armed_timeout_ms']}ms"
        )

        if bool(cfg["ramp_enabled"]):
            steps = cfg["ramp_steps"]
            if not steps:
                steps = [inspect_brightness]

            for b in steps:
                b = int(max(0, min(100, int(b))))
                self._set_light_brightness_checked(
                    light_id=light_id,
                    brightness=b,
                    action="SPOT_RAMP",
                )
                time.sleep(float(cfg["ramp_step_ms"]) / 1000.0)
        else:
            self._set_light_brightness_checked(
                light_id=light_id,
                brightness=inspect_brightness,
                action="SPOT_INSPECT_BRIGHTNESS",
            )

        if int(cfg["pre_ready_settle_ms"]) > 0:
            time.sleep(float(cfg["pre_ready_settle_ms"]) / 1000.0)

        self._flush_camera_frames(int(cfg["pre_ready_flush_frames"]))

        st.spot_armed = True
        st.spot_armed_ts = time.time()
        st.spot_armed_brightness = inspect_brightness
        st.status = f"{trigger} SPOT READY: {inspect_brightness}%"

        print(f"[LIGHT PREARM] ready brightness={inspect_brightness}%")
        return True

    def _spot_release(self, reason="release"):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            st.spot_armed = False
            return

        light_id = cfg["light_id"]
        idle_brightness = int(cfg["idle_brightness"])

        self._set_light_brightness_checked(
            light_id=light_id,
            brightness=idle_brightness,
            action="SPOT_IDLE_RESTORE",
        )

        st.spot_armed = False
        st.spot_armed_ts = 0.0
        st.spot_armed_brightness = 0

        print(f"[LIGHT PREARM] restored idle={idle_brightness}% reason={reason}")

    def _spot_timeout_tick(self):
        st = self.state
        cfg = self._get_spot_light_cfg()

        if not bool(cfg.get("enabled", False)):
            return

        if not bool(getattr(st, "spot_armed", False)):
            return

        timeout_ms = int(cfg.get("armed_timeout_ms", 3000))
        if timeout_ms <= 0:
            return

        elapsed_ms = int((time.time() - float(st.spot_armed_ts)) * 1000.0)

        if elapsed_ms > timeout_ms:
            try:
                self._spot_release(reason="timeout")
                st.status = "SPOT READY TIMEOUT"

            except Exception as e:
                st.spot_armed = False
                st.spot_armed_ts = 0.0
                st.spot_armed_brightness = 0

                if not st.light_error_latched:
                    st.light_error_latched = True

                    self._save_plc_error_event(
                        event_type="VISION_RUNTIME_ERROR",
                        error_code=21,
                        message="Light communication failed during spot timeout restore",
                        exception=e,
                    )

                self.plc.set_error(
                    code=21,
                    detail=str(e),
                )

                st.status = "LIGHT COMM ERROR: RESET REQUIRED"

    def _run_spot_inspect_once(self, frame_gray8, vis_bgr, avg5=False, trigger="PLC"):
        st = self.state
        cfg = self._get_spot_light_cfg()

        use_spot = bool(cfg.get("enabled", False))

        inspect_frame_gray8 = frame_gray8
        inspect_vis_bgr = vis_bgr

        try:
            if use_spot:
                if not bool(getattr(st, "spot_armed", False)):
                    self._spot_prearm(trigger=f"{trigger}_DIRECT")

                if int(cfg["inspect_settle_ms"]) > 0:
                    time.sleep(float(cfg["inspect_settle_ms"]) / 1000.0)

                g, v = self._flush_camera_frames(int(cfg["inspect_flush_frames"]))
                if g is not None and v is not None:
                    inspect_frame_gray8 = g
                    inspect_vis_bgr = v

            run_inspect_once(
                cam=self.cam,
                inspector=self.inspector,
                runtime_cfg=self.runtime_cfg,
                state=st,
                frame_gray8=inspect_frame_gray8,
                vis_bgr=inspect_vis_bgr,
                avg5=bool(avg5),
                use_cache=False,
                cache_every_n=1,
            )

            return st.last_overall_ok

        finally:
            if use_spot and bool(cfg.get("restore_after_inspect", True)):
                self._spot_release(reason=f"{trigger}_done")

    def _get_light_failures(self) -> Dict[str, str]:
        failures = {}

        try:
            light_state = self.light.get_state() or {}
        except Exception as e:
            return {
                "_controller": f"STATE_READ_FAILED:{e}"
            }

        for light_id, info in light_state.items():
            if not isinstance(info, dict):
                continue

            if info.get("ok") is False:
                failures[str(light_id)] = str(
                    info.get("reason", "UNKNOWN")
                )

        return failures
    
    def _get_vision_state_snapshot(self) -> Dict[str, Any]:
        camera_open = False

        try:
            cap = getattr(self.cam, "cap", None)
            camera_open = bool(cap is not None and cap.isOpened())
        except Exception:
            camera_open = False

        try:
            light_state = self.light.get_state()
        except Exception as e:
            light_state = {
                "read_failed": True,
                "error": str(e),
            }

        return {
            "app_status": str(self.state.status),
            "edit_mode": bool(self.state.edit_mode),
            "quit_requested": bool(self.state.quit_requested),

            "camera_open": camera_open,
            "camera_device": str(
                self.camera_info.get("device", "")
            ),

            "light_state": light_state,

            "last_overall_ok": self.state.last_overall_ok,
            "last_result_count": len(
                self.state.last_results or {}
            ),

            "tracking_stable": bool(
                self.state.tracking_stable
            ),
            "stable_frame_count": int(
                self.state.stable_frame_count
            ),

            "spot_armed": bool(self.state.spot_armed),
            "spot_armed_brightness": int(
                self.state.spot_armed_brightness
            ),

            "test_error_active": bool(
                self.state.test_error_active
            ),
            "test_error_type": str(
                self.state.test_error_type
            ),
            "test_error_code": int(
                self.state.test_error_code
            ),
            "test_error_injected_at": str(
                self.state.test_error_injected_at
            ),
            "test_error_request_id": str(
                self.state.test_error_request_id
            ),
            "test_error_safe_logical": bool(
                self.state.test_error_safe_logical
            ),
            "test_error_phase": str(
                self.state.test_error_phase
            ),
            "test_error_result": str(
                self.state.test_error_result
            ),
            "test_error_log_path": str(
                self.state.test_error_log_path
            ),
        }

    def _save_plc_error_event(
        self,
        event_type: str,
        error_code: int,
        message: str,
        exception: Optional[BaseException] = None,
    ):
        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {
                "snapshot_failed": True,
                "error": str(e),
            }

        saved_path = save_plc_error_log(
            logs_root=LOGS_ROOT,
            event_type=event_type,
            error_code=error_code,
            message=message,
            plc_snapshot=plc_snapshot,
            vision_snapshot=self._get_vision_state_snapshot(),
            exception=exception,
        )

        if saved_path:
            self.state.latest_error_log_path = str(saved_path)
            try:
                self.plc.trace_application_event(
                    event="ERROR_LOG_SAVED",
                    summary=(
                        f"code={int(error_code)} "
                        f"event_type={event_type} "
                        f"path={saved_path}"
                    ),
                    extra={
                        "error_code": int(error_code),
                        "error_event_type": str(event_type),
                        "error_log_path": str(saved_path),
                    },
                )
            except Exception:
                pass

        return saved_path

    def _poll_plc_comm_events(self):
        while True:
            try:
                event = self.plc.poll_comm_event()
            except AttributeError:
                return
            except Exception as e:
                print("[PLC] communication event read failed:", e)
                return

            if event is None:
                return

            error_code = int(
                event.get("error_code", 71)
            )
            message = str(
                event.get(
                    "message",
                    "PLC communication error",
                )
            )

            self._save_plc_error_event(
                event_type="PLC_COMMUNICATION_ERROR",
                error_code=error_code,
                message=message,
            )

            # 통신 복구 후 PLC에서 읽을 수 있도록 D201=3 유지
            self.plc.set_error(
                code=error_code,
                detail=message,
            )

            self.state.status = (
                f"PLC COMM ERROR {error_code}: RESET REQUIRED"
            )

    def _get_plc_error_test_cfg(self) -> Dict[str, Any]:
        cfg = self.plc_cfg.get("error_test", {}) or {}

        request_path = str(
            cfg.get(
                "request_path",
                "data/runtime/plc_error_test_request.json",
            )
        ).strip()

        if not os.path.isabs(request_path):
            request_path = os.path.join(
                PROJECT_ROOT,
                request_path,
            )

        allowed_types = {
            str(item).strip().lower()
            for item in (
                cfg.get(
                    "allowed_types",
                    [
                        "camera",
                        "light",
                        "inspection",
                        "plc_comm",
                    ],
                )
                or []
            )
            if str(item).strip()
        }

        return {
            "enabled": bool(cfg.get("enabled", False)),
            "request_path": os.path.abspath(request_path),
            "delete_after_read": bool(
                cfg.get("delete_after_read", True)
            ),
            "allowed_types": allowed_types,
            "safe_logical_mode": bool(cfg.get("safe_logical_mode", True)),
            "allow_hardware_fault_tests": bool(
                cfg.get("allow_hardware_fault_tests", False)
            ),
        }

    def _inject_plc_test_error(
        self,
        error_type: str,
        request_id: str,
        custom_message: str = "",
    ) -> bool:
        st = self.state
        error_type = str(error_type or "").strip().lower()
        test_cfg = self._get_plc_error_test_cfg()

        if not bool(test_cfg.get("enabled", False)):
            print("[PLC TEST] error injection rejected: test mode is disabled")
            return False

        definitions = {
            "camera": {
                "code": 11,
                "event_type": "PLC_TEST_CAMERA_ERROR",
                "message": (
                    "Forced camera error for PLC recovery test"
                ),
            },
            "light": {
                "code": 21,
                "event_type": "PLC_TEST_LIGHT_ERROR",
                "message": (
                    "Forced light error for PLC recovery test"
                ),
            },
            "inspection": {
                "code": 40,
                "event_type": "PLC_TEST_INSPECTION_ERROR",
                "message": (
                    "Forced inspection error for PLC recovery test"
                ),
            },
            "plc_comm": {
                "code": 71,
                "event_type": "PLC_TEST_COMM_ERROR",
                "message": (
                    "Forced PLC communication error state for "
                    "PLC recovery test; serial remains connected"
                ),
            },
        }

        definition = definitions.get(error_type)
        if definition is None:
            print(
                f"[PLC TEST] unsupported error type: "
                f"{error_type}"
            )
            return False

        try:
            current_status = int(
                self.plc.get_state_snapshot().get("status", 0)
            )
        except Exception:
            current_status = 0

        if st.test_error_active or current_status == 3:
            print(
                "[PLC TEST] error injection rejected: "
                "an error is already active; send D200=3 first"
            )
            return False

        error_code = int(definition["code"])
        message = str(custom_message or definition["message"])
        injected_at = time.strftime("%Y-%m-%d %H:%M:%S")

        safe_logical = bool(test_cfg.get("safe_logical_mode", True))

        # Service UI의 기본 강제 오류는 논리 오류만 발생시킨다.
        # 카메라/조명 파이프라인을 실제로 끊지 않으므로 반복 시험이 안전하다.
        if not safe_logical:
            if error_type == "camera":
                st.camera_error_latched = True
            elif error_type == "light":
                st.light_error_latched = True

        st.test_error_active = True
        st.test_error_type = error_type
        st.test_error_code = error_code
        st.test_error_injected_at = injected_at
        st.test_error_request_id = str(request_id)
        st.test_error_safe_logical = safe_logical
        st.test_error_phase = "WAITING_RESET"
        st.test_error_result = ""
        st.test_error_log_path = ""
        st.test_error_log_saved = False
        st.test_error_recovered_at = ""

        detail = (
            f"[TEST:{error_type}] {message} "
            f"request_id={request_id}"
        )

        # D201=3과 내부 error code를 먼저 반영한 뒤,
        # 동일 상태가 오류 JSON snapshot에도 저장되도록 한다.
        self.plc.set_error(
            code=error_code,
            detail=detail,
        )

        saved_path = self._save_plc_error_event(
            event_type=str(definition["event_type"]),
            error_code=error_code,
            message=detail,
            exception=RuntimeError(detail),
        )
        st.latest_error_log_path = str(saved_path or "")
        self._save_plc_test_event(
            phase="INJECTED",
            result="",
            message=detail,
            extra={
                "safe_logical_mode": bool(safe_logical),
                "error_log_path": str(saved_path or ""),
            },
        )

        try:
            self.plc.trace_application_event(
                event="TEST_ERROR_INJECTED",
                summary=(
                    f"type={error_type} code={error_code} "
                    f"request_id={request_id}"
                ),
                extra={
                    "test_error_type": error_type,
                    "test_error_code": error_code,
                    "test_request_id": request_id,
                    "error_log_path": str(saved_path or ""),
                },
            )
        except Exception:
            pass

        st.status = (
            f"TEST ERROR {error_code} "
            f"{error_type.upper()}: D200=3 RESET REQUIRED"
        )

        print(
            f"[PLC TEST] injected type={error_type} "
            f"code={error_code} log={saved_path}"
        )
        return True

    def _poll_plc_error_test_request(self):
        cfg = self._get_plc_error_test_cfg()
        if not cfg["enabled"]:
            return

        request_path = cfg["request_path"]
        if not os.path.exists(request_path):
            return

        request = None
        read_error = None

        try:
            with open(
                request_path,
                "r",
                encoding="utf-8-sig",
            ) as file:
                request = json.load(file)
        except Exception as e:
            read_error = e
        finally:
            if cfg["delete_after_read"]:
                try:
                    os.remove(request_path)
                except FileNotFoundError:
                    pass
                except Exception as e:
                    print(
                        f"[PLC TEST] request cleanup failed: {e}"
                    )

        if read_error is not None:
            print(
                f"[PLC TEST] invalid request file: {read_error}"
            )
            return

        if not isinstance(request, dict):
            print("[PLC TEST] request must be a JSON object")
            return

        request_id = str(
            request.get(
                "request_id",
                f"req-{time.time_ns()}",
            )
        )
        if request_id == self._plc_test_last_request_id:
            return

        self._plc_test_last_request_id = request_id

        error_type = str(
            request.get("type", "")
        ).strip().lower()

        if error_type not in cfg["allowed_types"]:
            print(
                f"[PLC TEST] type not allowed: {error_type}"
            )
            return

        self._inject_plc_test_error(
            error_type=error_type,
            request_id=request_id,
            custom_message=str(
                request.get("message", "") or ""
            ),
        )

    def _clear_spot_runtime_state(self):
        st = self.state
        st.spot_armed = False
        st.spot_armed_ts = 0.0
        st.spot_armed_brightness = 0

    def _clear_vision_runtime_state(self):
        st = self.state
        errors = []

        st.last_results = None
        st.last_overall_ok = None
        if hasattr(st, "last_overall_info"):
            st.last_overall_info = None

        st.pose_bad_cnt = 0
        st.tracking_stable = False
        st.stable_frame_count = 0
        st.last_auto_inspect_ts = 0.0

        self._clear_spot_runtime_state()

        try:
            mean_filters = getattr(self.inspector, "mean_filters", {}) or {}
            for mean_filter in mean_filters.values():
                mean_filter.reset()
            mean_filters.clear()
        except Exception as e:
            errors.append(f"mean filters: {e}")

        try:
            stabilizer_state = getattr(self.stabilizer, "state", None)
            if hasattr(stabilizer_state, "clear"):
                stabilizer_state.clear()
        except Exception as e:
            errors.append(f"stabilizer: {e}")

        try:
            aligner = getattr(self.inspector, "aligner", None)
            if aligner is not None:
                aligner.reset_templates()
            self._load_alignment_template()
        except Exception as e:
            errors.append(f"tracker: {e}")

        if errors:
            raise RuntimeError("; ".join(errors))

    def _get_plc_recovery_cfg(self):
        cfg = self.plc_cfg.get("recovery", {}) or {}
        return {
            "camera_open_timeout_sec": max(
                0.5,
                float(cfg.get("camera_open_timeout_sec", 2.0)),
            ),
            "camera_retry_interval_sec": max(
                0.02,
                float(cfg.get("camera_retry_interval_sec", 0.1)),
            ),
            "camera_grace_sec": max(
                0.0,
                float(cfg.get("camera_grace_sec", 2.0)),
            ),
            "frame_timeout_sec": max(
                0.1,
                float(cfg.get("frame_timeout_sec", 1.0)),
            ),
        }

    def _restart_camera_checked(self):
        recovery_cfg = self._get_plc_recovery_cfg()

        self.cam.release()
        time.sleep(0.2)
        self.cam.open()

        recovered_frame = None
        recovery_deadline = (
            time.time()
            + recovery_cfg["camera_open_timeout_sec"]
        )

        while time.time() < recovery_deadline:
            recovered_frame = self.cam.read()
            if recovered_frame is not None:
                break
            time.sleep(recovery_cfg["camera_retry_interval_sec"])

        if recovered_frame is None:
            raise RuntimeError("camera reopened but no frame was received")

        self.state.camera_error_grace_until = (
            time.time()
            + recovery_cfg["camera_grace_sec"]
        )
        return recovered_frame

    def _restart_light_checked(self):
        self.light.start()

        light_state = self.light.get_state() or {}
        self.runtime_cfg["_light_state"] = light_state

        failed_lights = {}
        for light_id, info in light_state.items():
            if not isinstance(info, dict):
                continue
            if info.get("ok") is False:
                failed_lights[str(light_id)] = str(
                    info.get("reason", "UNKNOWN")
                )

        if failed_lights:
            raise LightCommunicationError(
                f"light communication failed: {failed_lights}"
            )

        return light_state

    def _handle_plc_shutdown(self):
        st = self.state

        if bool(getattr(st, "plc_shutdown_started", False)):
            return

        st.plc_shutdown_started = True
        st.status = "PLC SHUTDOWN"
        self.plc.set_shutdown()  # D201=8 유지

        try:
            self._spot_release(reason="plc_shutdown")
        except Exception as e:
            print("[PLC] shutdown spot release failed:", e)

        try:
            self.light.stop()
        except Exception as e:
            print("[PLC] light stop before shutdown failed:", e)

        shutdown_cfg = self.plc_cfg.get("shutdown", {}) or {}
        hold_sec = float(shutdown_cfg.get("ready_hold_sec", 2.0))
        command = str(
            shutdown_cfg.get(
                "command",
                "sudo /sbin/shutdown -h now",
            )
        )

        if hold_sec > 0:
            time.sleep(hold_sec)

        try:
            shutdown_process = subprocess.Popen(command, shell=True)
            time.sleep(0.2)

            return_code = shutdown_process.poll()
            if return_code not in (None, 0):
                raise RuntimeError(
                    f"shutdown command exited with code {return_code}"
                )

            st.status = "PLC SHUTDOWN COMMAND SENT"
            st.quit_requested = True

        except Exception as e:
            print("[PLC] shutdown command failed:", e)

            self._save_plc_error_event(
                event_type="VISION_ERROR",
                error_code=90,
                message="Jetson shutdown command failed",
                exception=e,
            )

            self.plc.set_error(
                code=90,
                detail=str(e),
            )

            st.status = f"PLC SHUTDOWN ERROR: {e}"
            st.plc_shutdown_started = False

    def _handle_plc_emergency(self):
        st = self.state

        # 초기화 전에 현재 상태 저장
        # heartbeat, 내부 오류 코드, 마지막 검사 시각/소요시간 포함
        self._save_plc_error_event(
            event_type="EMERGENCY_STOP",
            error_code=51,
            message="D200=9 emergency stop received",
        )

        reset_ok = self._handle_plc_reset(
            reset_heartbeat=True,
        )

        if not reset_ok:
            # 복구 실패 원인은 _handle_plc_reset()에서 이미 로그 저장
            # D201=3 유지
            st.status = "PLC EMERGENCY: VISION ERROR"
            return

        # 정상 복구
        # D201=0, D202=0, D203=0
        st.status = "PLC EMERGENCY RESET: READY"

    def _handle_plc_reset(
        self,
        reset_heartbeat: bool = False,
    ):
        st = self.state

        try:
            before_state = self.plc.get_state_snapshot()
            current_status = int(before_state.get("status", 0))
            current_error_code = int(before_state.get("error_code", 0))
        except Exception:
            current_status = 0
            current_error_code = 0

        camera_error_codes = {10, 11}
        light_error_codes = {20, 21}
        plc_comm_error_codes = {70, 71}

        need_camera_restart = (
            st.camera_error_latched
            or current_error_code in camera_error_codes
        )
        need_light_restart = (
            st.light_error_latched
            or current_error_code in light_error_codes
        )

        # 강제 오류 UI의 기본 모드는 논리 시험이다.
        # Camera/Light 오류 코드를 사용해도 실제 하드웨어를 release/open 하지 않는다.
        if st.test_error_active and st.test_error_safe_logical:
            # 논리 테스트 자체는 하드웨어 재시작을 요구하지 않는다.
            # 단, 테스트 도중 실제 Camera/Light fault가 발생해 latch가 올라온 경우는 복구한다.
            need_camera_restart = bool(st.camera_error_latched)
            need_light_restart = bool(st.light_error_latched)

        # 이전 Reset 자체가 실패한 경우에는 하드웨어 전체 복구를 다시 시도한다.
        if current_status == 3 and current_error_code == 60:
            need_camera_restart = True
            need_light_restart = True

        # PLC 통신 오류는 비전 하드웨어 재시작 대상이 아니다.
        if current_error_code in plc_comm_error_codes:
            need_camera_restart = bool(st.camera_error_latched)
            need_light_restart = bool(st.light_error_latched)

        # D200=3 입력 즉시 D202를 0으로 초기화한다.
        # D200=9에서는 D203도 0으로 초기화한다.
        self.plc.prepare_reset(
            reset_heartbeat=reset_heartbeat
        )

        # 조명 오류가 이미 발생한 상태에서는 통신 명령을 먼저 보내지 않는다.
        # 조명 재시작 자체가 idle 밝기 복구를 수행한다.
        if need_light_restart:
            self._clear_spot_runtime_state()
        else:
            try:
                self._spot_release(reason="plc_reset")
            except Exception as e:
                need_light_restart = True
                st.light_error_latched = True
                self._clear_spot_runtime_state()

                self._save_plc_error_event(
                    event_type="VISION_RESET_RECOVERY",
                    error_code=21,
                    message=(
                        "Light idle restore failed during PLC reset; "
                        "light controller restart will be attempted"
                    ),
                    exception=e,
                )

        try:
            self._clear_vision_runtime_state()
        except Exception as e:
            self._save_plc_error_event(
                event_type="VISION_RESET_ERROR",
                error_code=60,
                message="Vision runtime state reset failed",
                exception=e,
            )
            self.plc.set_error(code=60, detail=str(e))
            st.status = "PLC RESET ERROR: RUNTIME"
            if st.test_error_active:
                st.test_error_phase = "RECOVERY_FAILED"
                st.test_error_result = "FAIL"
                self._save_plc_test_event(
                    phase="RECOVERY_FAILED",
                    result="FAIL",
                    message=f"Vision runtime reset failed: {e}",
                )
            return False

        if need_camera_restart:
            try:
                self._restart_camera_checked()
                st.camera_error_latched = False
            except Exception as e:
                st.camera_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RESET_ERROR",
                    error_code=10,
                    message="Camera restart failed during PLC reset",
                    exception=e,
                )
                self.plc.set_error(code=10, detail=str(e))
                st.status = "PLC RESET ERROR: CAMERA"
                if st.test_error_active:
                    st.test_error_phase = "RECOVERY_FAILED"
                    st.test_error_result = "FAIL"
                    self._save_plc_test_event(
                        phase="RECOVERY_FAILED",
                        result="FAIL",
                        message=f"Camera recovery failed: {e}",
                    )
                return False

        if need_light_restart:
            try:
                self._restart_light_checked()
                st.light_error_latched = False
            except Exception as e:
                st.light_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RESET_ERROR",
                    error_code=21,
                    message="Light restart failed during PLC reset",
                    exception=e,
                )
                self.plc.set_error(code=21, detail=str(e))
                st.status = "PLC RESET ERROR: LIGHT"
                if st.test_error_active:
                    st.test_error_phase = "RECOVERY_FAILED"
                    st.test_error_result = "FAIL"
                    self._save_plc_test_event(
                        phase="RECOVERY_FAILED",
                        result="FAIL",
                        message=f"Light recovery failed: {e}",
                    )
                return False

        recovered_test_type = str(st.test_error_type)
        recovered_test_code = int(st.test_error_code)
        recovered_test_id = str(st.test_error_request_id)

        st.camera_error_latched = False
        st.light_error_latched = False
        st.test_error_active = False

        self.plc.reset_to_ready(
            reset_heartbeat=reset_heartbeat
        )

        if recovered_test_type:
            st.test_error_phase = "RECOVERED"
            st.test_error_result = "PASS"
            st.test_error_recovered_at = time.strftime("%Y-%m-%d %H:%M:%S")
            self._save_plc_test_event(
                phase="RECOVERED",
                result="PASS",
                message=(
                    f"D200=3 recovery completed for {recovered_test_type} "
                    f"code={recovered_test_code} request_id={recovered_test_id}"
                ),
                extra={
                    "reset_heartbeat": bool(reset_heartbeat),
                    "safe_logical_mode": bool(st.test_error_safe_logical),
                },
            )
            try:
                self.plc.trace_application_event(
                    event="TEST_ERROR_RECOVERED",
                    summary=(
                        f"type={recovered_test_type} "
                        f"code={recovered_test_code} "
                        f"reset_heartbeat={bool(reset_heartbeat)}"
                    ),
                    extra={
                        "test_error_type": recovered_test_type,
                        "test_error_code": recovered_test_code,
                        "recovery_ok": True,
                    },
                )
            except Exception:
                pass

        st.status = "PLC RESET: READY"
        return True

    def _log_plc_command_sequence_error(
        self,
        command_name: str,
        current_status: int,
        reason: str,
    ):
        status_names = {
            0: "READY",
            1: "BUSY",
            2: "DONE",
            3: "VISION_ERROR",
            8: "SHUTDOWN",
        }

        message = (
            f"PLC command sequence rejected: "
            f"command={command_name}, "
            f"status={current_status}"
            f"({status_names.get(current_status, 'UNKNOWN')}), "
            f"reason={reason}"
        )

        self._save_plc_error_event(
            event_type="PLC_COMMAND_SEQUENCE_ERROR",
            error_code=50,
            message=message,
        )

        print(f"[PLC] {message}")

    def _run_plc_inspect_tick(self, frame_gray8, vis_bgr):
        st = self.state

        cmd = self.plc.poll_command()
        if cmd is None:
            return

        if cmd == "reset":
            self._handle_plc_reset()
            return

        if cmd == "shutdown":
            self._handle_plc_shutdown()
            return

        if cmd == "emergency":
            self._handle_plc_emergency()
            return

        try:
            plc_snapshot = self.plc.get_state_snapshot()
            current_status = int(
                plc_snapshot.get("status", 0)
            )
        except Exception:
            current_status = 0

        # 종료가 시작된 상태에서는 다른 명령을 처리하지 않는다.
        if current_status == 8:
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision shutdown is already in progress",
            )
            return

        # 검사 결과가 남아 있는 동안에는 D200=3만 정상 초기화 명령이다.
        if current_status == 2 and cmd in ("prepare", "inspect"):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="previous inspection result has not been reset by D200=3",
            )

            # D201=2, D202=OK/NG 그대로 유지
            st.status = "PLC COMMAND REJECTED: RESULT RESET REQUIRED"
            return

        # 비전 오류 상태에서는 Reset, 종료, 비상정지만 허용한다.
        if current_status == 3 and cmd in ("prepare", "inspect"):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision error must be recovered by D200=3",
            )

            # D201=3 유지
            st.status = "PLC COMMAND REJECTED: VISION RESET REQUIRED"
            return

        # Busy 중 중복 명령 차단
        if current_status == 1 and cmd in ("prepare", "inspect"):
            self._log_plc_command_sequence_error(
                command_name=str(cmd),
                current_status=current_status,
                reason="vision is already processing another command",
            )

            # D201=1 유지
            st.status = "PLC COMMAND REJECTED: VISION BUSY"
            return
        
        if st.camera_error_latched or st.light_error_latched:
            st.status = "PLC COMMAND REJECTED: VISION ERROR"
            return
        
        if cmd == "command_error":
            try:
                snapshot = self.plc.get_state_snapshot()
                command_value = int(
                    snapshot.get("command", 0)
                )
                current_status = int(
                    snapshot.get("status", 0)
                )
            except Exception:
                command_value = -1
                current_status = 0

            self._log_plc_command_sequence_error(
                command_name=f"UNKNOWN_D200_{command_value}",
                current_status=current_status,
                reason="unsupported D200 command value",
            )

            # PLC 측 잘못된 값이므로 D201=3으로 변경하지 않는다.
            st.status = (
                f"PLC UNKNOWN COMMAND: D200={command_value}"
            )
            return

        if cmd == "prepare":
            if st.edit_mode:
                self._log_plc_command_sequence_error(
                    command_name="prepare",
                    current_status=current_status,
                    reason="vision is in EDIT mode",
                )
                st.status = "PLC PREPARE REJECTED: EDIT MODE"
                return

            if frame_gray8 is None or vis_bgr is None:
                error = RuntimeError("camera frame is unavailable")

                st.camera_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=11,
                    message="D200=1 received but camera frame is unavailable",
                    exception=error,
                )

                self.plc.set_error(
                    code=11,
                    detail=str(error),
                )

                st.status = "PLC PREPARE ERROR: NO CAMERA FRAME"
                return

            try:
                self.plc.set_busy()
                st.status = "PLC PREPARE BUSY"

                armed = self._spot_prearm(trigger="PLC")

                # 준비 완료 후 Ready
                # D202 값은 변경하지 않음
                self.plc.set_idle()

                st.status = (
                    "PLC READY: SPOT ARMED"
                    if armed
                    else "PLC READY"
                )

            except LightCommunicationError as e:
                st.light_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=21,
                    message="Light communication failed during PLC prepare",
                    exception=e,
                )

                self.plc.set_error(
                    code=21,
                    detail=str(e),
                )

                st.status = "PLC PREPARE LIGHT ERROR: RESET REQUIRED"

            except Exception as e:
                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=40,
                    message="PLC inspection preparation failed",
                    exception=e,
                )

                self.plc.set_error(
                    code=40,
                    detail=str(e),
                )

                st.status = "PLC PREPARE ERROR: RESET REQUIRED"

            return

        if cmd != "inspect":
            return

        if st.edit_mode:
            self._log_plc_command_sequence_error(
                command_name="inspect",
                current_status=current_status,
                reason="vision is in EDIT mode",
            )

            st.status = "PLC INSPECT REJECTED: EDIT MODE"
            return

        if frame_gray8 is None or vis_bgr is None:
            error = RuntimeError("camera frame is unavailable")

            st.camera_error_latched = True

            self._save_plc_error_event(
                event_type="VISION_RUNTIME_ERROR",
                error_code=11,
                message="D200=2 received but camera frame is unavailable",
                exception=error,
            )

            self.plc.set_error(
                code=11,
                detail=str(error),
            )

            st.status = "PLC INSPECT ERROR: NO CAMERA FRAME"
            return

        try:
            self.plc.set_busy()
            st.status = "PLC INSPECT BUSY"

            # 이전 내부 검사 결과 제거
            # PLC D202 값은 변경하지 않음
            st.last_overall_ok = None
            st.last_results = None

            started_at = time.perf_counter()

            overall_ok = self._run_spot_inspect_once(
                frame_gray8,
                vis_bgr,
                avg5=bool(
                    self.runtime_cfg.get(
                        "plc_inspect_avg5",
                        False,
                    )
                ),
                trigger="PLC",
            )

            elapsed_ms = int(
                (time.perf_counter() - started_at) * 1000.0
            )

            results = st.last_results

            if overall_ok is None:
                raise InspectionResultError(
                    "inspection overall result was not generated"
                )

            if not isinstance(results, dict) or not results:
                raise InspectionResultError(
                    "inspection ROI results were not generated"
                )

            invalid_roi_ids = []

            for roi_id, result in results.items():
                if isinstance(result, dict):
                    roi_ok = result.get("ok")
                else:
                    roi_ok = getattr(result, "ok", None)

                if roi_ok is None:
                    invalid_roi_ids.append(str(roi_id))

            if invalid_roi_ids:
                raise InspectionResultError(
                    "ROI result has no OK/NG value: "
                    + ",".join(invalid_roi_ids)
                )

            self.plc.set_done(
                ok=bool(overall_ok),
                elapsed_ms=elapsed_ms,
            )

            st.status = (
                "PLC INSPECT DONE: OK"
                if overall_ok
                else "PLC INSPECT DONE: NG"
            )

        except LightCommunicationError as e:
            st.light_error_latched = True

            self._save_plc_error_event(
                event_type="VISION_RUNTIME_ERROR",
                error_code=21,
                message="Light communication failed during inspection",
                exception=e,
            )

            self.plc.set_error(
                code=21,
                detail=str(e),
            )

            st.status = "PLC INSPECT LIGHT ERROR: RESET REQUIRED"

        except InspectionResultError as e:
            self._save_plc_error_event(
                event_type="VISION_INSPECTION_ERROR",
                error_code=40,
                message="Inspection result generation failed",
                exception=e,
            )

            self.plc.set_error(
                code=40,
                detail=str(e),
            )

            st.status = "PLC INSPECT RESULT ERROR: RESET REQUIRED"

        except Exception as e:
            print("[PLC] inspect exception:", e)

            self._save_plc_error_event(
                event_type="VISION_INSPECTION_ERROR",
                error_code=41,
                message="Unexpected exception during inspection",
                exception=e,
            )

            self.plc.set_error(
                code=41,
                detail=str(e),
            )

            st.status = "PLC INSPECT EXCEPTION: RESET REQUIRED"

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
                show_metrics=bool(self.runtime_cfg.get("dev_mode", False))
                and bool(self.runtime_cfg.get("dev_overlay_metrics", False)),
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
                show_metrics=bool(self.runtime_cfg.get("dev_mode", False))
                and bool(self.runtime_cfg.get("dev_overlay_metrics", False)),
            )
            
    def _handle_key_input(self, key, frame_gray8, vis_bgr):
        st = self.state

        if self.service_panel.handle_key(key):
            return

        if key in (ord('v'), ord('V')):
            self._toggle_service_panel()
            return

        if key not in (-1, 255):
            print(f"[DBG KEY] raw={key} chr={repr(chr(key)) if 32 <= key <= 126 else 'NONPRINT'}")

        if key in (ord('d'), ord('D')):   # delete OK/NG Dataset reset
            print("[DBG KEY] DATASET RESET HOTKEY")
            self._reset_dataset()
            return

        if key in (ord('b'), ord('B')):   # baseline_profile.json reset
            print("[DBG KEY] BASELINE RESET HOTKEY")
            self._reset_baseline()
            return
        
        if key in (ord('u'), ord('U')):
            print("[DBG KEY] BASELINE UPDATE HOTKEY")
            if self._update_baseline_from_ok_results():
                st.status = "Baseline updated from OK result"
            else:
                st.status = "Baseline update skipped"
            return
        
        if key in (ord('l'), ord('L')):
            self._handle_baseline_key(frame_gray8)
            print("[DBG KEY] BASELINE LEARN HOTKEY")
            return
        
        if (not st.edit_mode) and key in (ord('t'), ord('T')):
            self._save_roi_and_template(frame_gray8)
            return
        
        consumed, sample_msg = handle_sample_keys(
            key,
            frame_gray8,
            vis_bgr,
            edit_mode=st.edit_mode,
            roi_mgr=self.roi_mgr,
            data_dir=DATA_DIR,
            snapshot_keep=int(self.runtime_cfg.get("snapshot_keep", 10)),
            last_results=st.last_results,
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

    def _add_baseline_from_last_results(self):
        st = self.state

        trk_score = getattr(self.state, "trk_score", 1.0)
        if trk_score < 0.90:
            self.state.status = f"Baseline skipped: low track {trk_score:.3f}"
            print(f"[AUTO BASELINE] {self.state.status}")
            return False

        if not st.last_results:
            st.status = "No inspect result to learn"
            print(f"[AUTO BASELINE] results type={type(st.last_results)}")
            return False

        for roi_id, res in st.last_results.items():
            metrics = getattr(res, "metrics", None) or {}
            roi_name = f"ROI{roi_id}"

            if roi_name in ("ROI2", "ROI3", "ROI4", "ROI5"):
                v = metrics.get("dark_ratio", None)
                if v is not None:
                    self.baseline.add_sample(roi_name, "dark_ratio", v)

            elif roi_name == "ROI6":
                v = metrics.get("blob_count", None)
                if v is None:
                    v = metrics.get("blob", None)
                if v is not None:
                    self.baseline.add_sample(roi_name, "blob_count", v)

        st.baseline_count += 1

        if st.baseline_count >= st.baseline_target_count:
            self.baseline.save()
            st.status = f"Baseline saved ({st.baseline_count}/{st.baseline_target_count})"
            print(f"[AUTO BASELINE] {st.status} -> {self.baseline_path}")
            st.baseline_learning = False
            st.baseline_count = 0
            self.baseline = AutoBaseline(self.baseline_path)
        else:
            st.status = f"Baseline sample added ({st.baseline_count}/{st.baseline_target_count})"

        return True

    def _handle_baseline_key(self, frame_gray8=None):
        st = self.state

        if st.edit_mode:
            st.status = "Baseline learn only in RUN mode"
            return

        if frame_gray8 is None:
            st.status = "No frame for baseline"
            return

        try:
            auto_mode_backup = getattr(self.inspector, "auto_mode", True)
            self.inspector.auto_mode = False

            try:
                st.last_overall_ok, st.last_results = self.inspector.inspect(
                    frame_gray8,
                    auto_mode=False
                )
            finally:
                self.inspector.auto_mode = auto_mode_backup
        except Exception as e:
            st.status = f"Baseline inspect fail: {e}"
            return

        if not st.baseline_learning:
            st.baseline_learning = True
            st.baseline_target_count = int(self.runtime_cfg.get("baseline_ok_count", 10))
            st.baseline_count = 0
            self.baseline = AutoBaseline(self.baseline_path)

        self._add_baseline_from_last_results()

    def _reset_dataset(self):
        import shutil

        targets = [
            os.path.join(DATA_DIR, "dataset", "OK"),
            os.path.join(DATA_DIR, "dataset", "NG"),
            os.path.join(DATA_DIR, "templates"),
        ]
        print("[DBG RESET] dataset targets =", targets)

        for p in targets:
            os.makedirs(p, exist_ok=True)
            for name in os.listdir(p):
                fp = os.path.join(p, name)
                print("[DBG RESET] remove:", fp)
                try:
                    if os.path.isfile(fp) or os.path.islink(fp):
                        os.remove(fp)
                    elif os.path.isdir(fp):
                        shutil.rmtree(fp)
                except Exception as e:
                    print("[DBG RESET] fail:", fp, e)

        self.state.status = "DATASET RESET DONE"
        print("[DBG RESET] DATASET RESET DONE")

    def _reset_baseline(self):
        try:
            print("[DBG RESET] baseline path =", self.baseline_path)
            if os.path.exists(self.baseline_path):
                os.remove(self.baseline_path)
                print("[DBG RESET] baseline removed")
            else:
                print("[DBG RESET] baseline file not found")

            self.baseline = AutoBaseline(self.baseline_path)
            self.state.baseline_learning = False
            self.state.baseline_count = 0
            self.state.status = "BASELINE RESET DONE"
            print("[DBG RESET] BASELINE RESET DONE")
        except Exception as e:
            print("[DBG RESET] BASELINE RESET FAIL:", e)
            self.state.status = f"BASELINE RESET FAIL: {e}"

    def _draw_ui(self, vis):
        st = self.state
        st.active_roi_id = self.roi_mgr.selected_id

        overlay.draw_status_bar(vis, st.status)

        if (not st.edit_mode) and (st.last_overall_ok is not None):
            overlay.draw_overall_banner(vis,st.last_overall_ok,info=getattr(st, "last_overall_info", None),)

        st.last_buttons = render_control_bar(
            vis,
            st.edit_mode,
            show_service=bool(self.service_panel.enabled),
        )

        if bool(self.runtime_cfg.get("enable_pose_guide", True)):
            vis = draw_pose_message(
                vis,
                st.pose_bad_cnt,
                int(self.runtime_cfg.get("pose_bad_n", 5)),
            )

        draw_mode_indicator(vis, st.edit_mode)
        draw_dev_hud(vis, st, self.product_profile)

        try:
            plc_snapshot = self.plc.get_state_snapshot()
        except Exception as e:
            plc_snapshot = {
                "serial_open": False,
                "comm_fault_active": True,
                "error_code": 71,
                "error_detail": str(e),
            }

        try:
            recent_events = self.plc.get_recent_events(24)
        except Exception:
            recent_events = []

        try:
            roi_debug = self.inspector.get_debug_grid()
        except Exception:
            roi_debug = None

        vis = self.service_panel.draw(
            vis,
            plc_snapshot=plc_snapshot,
            recent_events=recent_events,
            app_state=st,
            roi_debug=roi_debug,
            latest_log_path=st.test_error_log_path,
            error_log_path=st.latest_error_log_path,
            test_summary=self._get_service_test_summary(),
        )
        return vis
    
    def _prepare_frame(self, frame):
        st = self.state

        if frame is None:
            return None, None

        if frame.ndim == 3:
            # B0429 grayscale path usually arrives as 3-channel gray/BGR.
            # B0251/IMX477 nvargus path arrives as real BGR color, so convert to proper gray.
            if frame.shape[2] >= 3:
                frame_gray8 = cv2.cvtColor(frame[:, :, :3], cv2.COLOR_BGR2GRAY)
            else:
                frame_gray8 = frame[:, :, 0]
        else:
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
        try:
            return self.cam.read()

        except Exception as e:
            st = self.state

            if not st.camera_error_latched:
                st.camera_error_latched = True

                self._save_plc_error_event(
                    event_type="VISION_RUNTIME_ERROR",
                    error_code=11,
                    message="Camera frame read raised an exception",
                    exception=e,
                )

                self.plc.set_error(
                    code=11,
                    detail=str(e),
                )

            st.status = "CAMERA READ ERROR: RESET REQUIRED"
            return None
    # -------------------------
    # Main loop
    # -------------------------
    def run(self):
        st = self.state
        self.plc.start()

        plc_required = bool(
            self.plc_cfg.get("enabled", False)
        )

        plc_start_error = (
            plc_required
            and not self.plc.is_connected()
        )

        startup_error = plc_start_error

        if plc_required and not plc_start_error:
            self.plc.set_busy()

        # 카메라 초기화
        try:
            self.cam.open()
            st.camera_error_latched = False
            recovery_cfg = self._get_plc_recovery_cfg()
            st.camera_error_grace_until = (
                time.time()
                + recovery_cfg["camera_grace_sec"]
            )

        except Exception as e:
            startup_error = True
            st.camera_error_latched = True
            st.status = f"CAMERA INIT ERROR: {e}"

            self._save_plc_error_event(
                event_type="VISION_STARTUP_ERROR",
                error_code=10,
                message="Camera initialization failed",
                exception=e,
            )

            self.plc.set_error(
                code=10,
                detail=str(e),
            )

        if plc_start_error:
            self._poll_plc_comm_events()

        # 조명 초기화
        try:
            self.light.start()
            self.runtime_cfg["_light_state"] = self.light.get_state()

            light_failures = self._get_light_failures()

            if light_failures:
                raise RuntimeError(
                    f"light startup failed: {light_failures}"
                )

            st.light_error_latched = False

        except Exception as e:
            startup_error = True
            st.light_error_latched = True
            st.status = f"LIGHT START ERROR: {e}"

            self._save_plc_error_event(
                event_type="VISION_STARTUP_ERROR",
                error_code=20,
                message="Light initialization failed",
                exception=e,
            )

            self.plc.set_error(
                code=20,
                detail=str(e),
            )

        # 시작 과정 중 발생한 PLC 통신 이벤트를 반영한 뒤 Ready를 결정한다.
        self._poll_plc_comm_events()

        if plc_required and not self.plc.is_connected():
            startup_error = True

        try:
            plc_status = int(
                self.plc.get_state_snapshot().get("status", 0)
            )
            if plc_required and plc_status == 3:
                startup_error = True
        except Exception:
            if plc_required:
                startup_error = True

        if not startup_error:
            self.plc.reset_to_ready(reset_heartbeat=False)
            st.status = "PLC READY" if plc_required else "READY"

        test_cfg = self._get_plc_error_test_cfg()
        if test_cfg["enabled"]:
            print(
                "[PLC TEST] forced-error test mode enabled "
                f"request_path={test_cfg['request_path']}"
            )
            try:
                self.plc.trace_application_event(
                    event="ERROR_TEST_MODE_ENABLED",
                    summary=(
                        "forced-error test mode enabled "
                        f"request_path={test_cfg['request_path']}"
                    ),
                    extra={
                        "test_mode_enabled": True,
                        "test_request_path": test_cfg["request_path"],
                    },
                )
            except Exception:
                pass

        recovery_cfg = self._get_plc_recovery_cfg()
        frame_timeout_sec = recovery_cfg["frame_timeout_sec"]
        last_ok_frame_time = time.time()
        last_camera_vis = None
        last_gray_frame = None

        while True:
            self._poll_plc_comm_events()
            self._poll_plc_error_test_request()

            frame = self._read_frame()

            if frame is None:
                now = time.time()
                no_frame_sec = now - last_ok_frame_time

                # 영상이 없어도 PLC 명령과 조명 타임아웃은 계속 처리
                self._run_plc_inspect_tick(None, None)
                self._spot_timeout_tick()

                if (
                    no_frame_sec > frame_timeout_sec
                    and now >= st.camera_error_grace_until
                    and not st.camera_error_latched
                ):
                    error = RuntimeError(
                        f"camera frame unavailable for {no_frame_sec:.2f} sec"
                    )

                    st.camera_error_latched = True

                    self._save_plc_error_event(
                        event_type="VISION_RUNTIME_ERROR",
                        error_code=11,
                        message="Camera frame reception stopped",
                        exception=error,
                    )

                    self.plc.set_error(
                        code=11,
                        detail=str(error),
                    )

                    st.status = "CAMERA FRAME ERROR: RESET REQUIRED"

                elif (
                    no_frame_sec > frame_timeout_sec
                    and not st.camera_error_latched
                ):
                    st.status = "WAITING FOR CAMERA FRAME"

                if last_camera_vis is not None:
                    no_frame_vis = last_camera_vis.copy()
                else:
                    no_frame_vis = np.zeros(
                        (self.frame_height, self.frame_width, 3),
                        dtype=np.uint8,
                    )

                warning_layer = no_frame_vis.copy()
                cv2.rectangle(
                    warning_layer,
                    (0, 0),
                    (self.frame_width, 72),
                    (0, 0, 180),
                    -1,
                )
                cv2.addWeighted(
                    warning_layer,
                    0.72,
                    no_frame_vis,
                    0.28,
                    0,
                    no_frame_vis,
                )
                cv2.putText(
                    no_frame_vis,
                    "CAMERA FRAME UNAVAILABLE - PLC RESET / SERVICE CHECK",
                    (24, 47),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.85,
                    (255, 255, 255),
                    2,
                    cv2.LINE_AA,
                )

                no_frame_vis = self._draw_ui(no_frame_vis)
                cv2.imshow(self.win, no_frame_vis)

                try:
                    key = cv2.waitKeyEx(1)
                except Exception:
                    key = cv2.waitKey(1)

                # 카메라가 없어도 SERVICE 패널/종료 키는 계속 동작한다.
                self._handle_key_input(key, None, no_frame_vis)

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

            last_gray_frame = frame_gray8.copy()
            last_camera_vis = vis.copy()

            self._run_plc_inspect_tick(frame_gray8, vis)
            self._spot_timeout_tick()

            if (
                not st.edit_mode
                and not st.camera_error_latched
                and not st.light_error_latched
                and not st.test_error_active
            ):
                self._run_auto_inspect_tick(frame_gray8, vis)
        
            # Status + banner + control Bar(button) + HUD  Draw
            vis = self._draw_ui(vis)

            # show
            cv2.imshow(self.win, vis)

            # key
            try:
                key = cv2.waitKeyEx(1)
            except Exception:
                key = cv2.waitKey(1)

            self._handle_key_input(key, frame_gray8, vis)

            if st.quit_requested:
                break

        try:
            cv2.destroyWindow(self.win)
            cv2.waitKey(1)
        except Exception:
            pass

        try:
            self.plc.stop()
        except Exception as e:
            print("[PLC] stop failed:", e)

        try:
            self.light.stop()
        except Exception as e:
            print("[LIGHT] stop failed:", e)

        # B0251 / nvargus는 release()에서 Argus cleanup 대기 때문에
        # 종료가 4~5초 지연될 수 있음.
        fast_exit = False
        try:
            pipeline_type = str(self.camera_info.get("pipeline_type", "") or "")
            fast_exit = pipeline_type == "nvargus_bgr"
        except Exception:
            fast_exit = False

        if fast_exit:
            print("[CAM] fast exit: skip blocking nvargus release")
            try:
                cv2.destroyAllWindows()
            except Exception:
                pass
            sys.stdout.flush()
            sys.stderr.flush()
            os._exit(0)

        try:
            self.cam.release()
        except Exception as e:
            print("[CAM] release failed:", e)

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass


def main():
    app = None

    try:
        app = VisionApp()
        app.run()

    except Exception as e:
        print(f"[FATAL] vision application terminated: {e}")

        if app is not None:
            try:
                app._save_plc_error_event(
                    event_type="VISION_FATAL_ERROR",
                    error_code=41,
                    message="Unhandled vision application exception",
                    exception=e,
                )
            except Exception:
                pass

            try:
                app.plc.set_error(code=41, detail=str(e))
            except Exception:
                pass

            try:
                app.plc.stop()
            except Exception:
                pass

            try:
                app.light.stop()
            except Exception:
                pass

            try:
                app.cam.release()
            except Exception:
                pass

        try:
            cv2.destroyAllWindows()
        except Exception:
            pass

        raise


if __name__ == "__main__":
    main()
