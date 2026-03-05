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

from ui.overlay import draw_control_bar
from ui import overlay_clean as overlay  # production overlay
from typing import Optional, Dict, Any
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


BUTTON_TO_CMD = {
    "toggle_edit": UICmd.TOGGLE_MODE,
    "inspect": UICmd.INSPECT,
    "autotune": UICmd.AUTOTUNE,
    "reload": UICmd.RELOAD,
    "save": UICmd.SAVE,
    "next": UICmd.NEXT,
    "nxt": UICmd.NEXT,
    "clear": UICmd.CLEAR,
    "delete": UICmd.DELETE,
    "quit": UICmd.QUIT,
    "autoinspect": UICmd.TOGGLE_AUTO_INSPECT,  # optional button
}


# =========================
# Config
# =========================
DEV_MODE = True

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
DATA_DIR     = os.path.join(PROJECT_ROOT, "data")
ROI_DIR      = os.path.join(DATA_DIR, "roi")
ROI_PATH     = os.path.join(ROI_DIR, "roi.json")
RECIPE_PATH  = os.path.join(ROI_DIR, "recipe_static.json")
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

NORMALIZE_ENABLED = True
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

def roi_label_pos(x, y, w, h, margin=25):
    tx = x
    ty = y - margin
    if ty < 18:
        ty = y + h + 18
    return int(tx), int(ty)

def prune_snapshots(path, max_keep=200):
    try:
        files = []
        for fn in os.listdir(path):
            if fn.endswith(".png") or fn.endswith(".jpg"):
                p = os.path.join(path, fn)
                files.append((os.path.getmtime(p), p))
        files.sort(reverse=True)
        for _, p in files[max_keep:]:
            try: os.remove(p)
            except Exception: pass
    except Exception:
        pass

def prune_manifests(dir_path, keep=200):
    try:
        items = []
        for fn in os.listdir(dir_path):
            if fn.endswith(".json"):
                p = os.path.join(dir_path, fn)
                items.append((os.path.getmtime(p), p))
        items.sort(reverse=True)

        for _, jpath in items[keep:]:
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for k in ("raw", "overlay", "crop"):
                    p = meta.get(k)
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
                os.remove(jpath)
            except Exception:
                try: os.remove(jpath)
                except Exception: pass
    except Exception:
        pass

def _get_roi_by_id(roi_mgr, roi_id: int):
    try:
        for r in getattr(roi_mgr, "rois", []):
            if int(r.get("id", -1)) == int(roi_id):
                return r
    except Exception:
        pass
    return None

def _crop_roi(gray8, roi_mgr, roi_id: int):
    r = _get_roi_by_id(roi_mgr, roi_id)
    if r is None or gray8 is None:
        return None
    x = int(r.get("x", 0)); y = int(r.get("y", 0)); w = int(r.get("w", 0)); h = int(r.get("h", 0))
    if w <= 0 or h <= 0:
        return None
    H, W = gray8.shape[:2]
    x = max(0, min(W-1, x)); y = max(0, min(H-1, y))
    x2 = max(0, min(W, x+w)); y2 = max(0, min(H, y+h))
    if x2 <= x or y2 <= y:
        return None
    return gray8[y:y2, x:x2].copy()

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
    last_overall_ok: Optional[bool[str, Any]] = None

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

        # load saved tracker template if exists
        self._load_alignment_template()

        self.win = "Static Mode - ROI Setup"
        cv2.namedWindow(self.win, cv2.WINDOW_NORMAL)
        cv2.resizeWindow(self.win, WIDTH, HEIGHT)
        cv2.setMouseCallback(self.win, self._mouse_router)

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
                        st.pending_cmd = BUTTON_TO_CMD.get(bid, UICmd.NONE)
                    return

            if st.edit_mode:
                self.editor._on_mouse(event, x, y, flags, None)
            return

        if st.edit_mode:
            self.editor._on_mouse(event, x, y, flags, None)

    def _key_to_cmd(self, key: int) -> UICmd:
        keymap = {
            27: UICmd.QUIT, ord('q'): UICmd.QUIT,
            ord('e'): UICmd.TOGGLE_MODE,
            ord('s'): UICmd.SAVE,
            ord('n'): UICmd.NEXT,
            ord('r'): UICmd.CLEAR,
            ord('p'): UICmd.RELOAD,
            ord('c'): UICmd.AUTOTUNE,
            32: UICmd.INSPECT,
            ord('x'): UICmd.DELETE,
            ord('a'): UICmd.TOGGLE_AUTO_INSPECT,  # NEW
        }
        return keymap.get(key, UICmd.NONE)

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

    def _inspect_once(self, frame_gray8, vis_bgr):
        """
        Single inspect execution.
        Updates: last_results, last_overall_ok, pose_bad_cnt
        """
        st = self.state

        # 5-frame avg (same as your current behavior)
        frames = []
        for _ in range(5):
            f = self.cam.read()
            if f is None:
                continue
            if getattr(f, "ndim", 0) == 3:
                f = f[:, :, 0]
            frames.append(f)
        avg = np.mean(frames, axis=0).astype("uint8") if frames else frame_gray8

        overall_ok, results = self.inspector.inspect(avg)

        # persist
        try:
            self.inspector.save_run(avg, vis_bgr.copy(), overall_ok, results)
        except Exception as e:
            print("[DBG] save_run failed:", e)

        st.last_results = {str(k): v for k, v in results.items()} if results else {}
        st.last_overall_ok = overall_ok
        st.status = f"INSPECT {'OK' if overall_ok else 'NG'}"
        try:
            self.inspector.log_result(st.last_overall_ok, st.last_results)
        except Exception:
            pass

        # pose counter update (always from last_results)
        r = (st.last_results or {}).get(POSE_ROI_ID_STR)
        bc = None
        if r is not None and hasattr(r, "metrics"):
            bc = (r.metrics or {}).get(POSE_METRIC_KEY, None)
        elif isinstance(r, dict):
            bc = (r.get("metrics") or {}).get(POSE_METRIC_KEY, None)

        if bc is None:
            # metric 없으면 카운트 유지(혹은 리셋)
            st.pose_bad_cnt = 0
        else:
            if int(bc) == int(POSE_EXPECT):
                st.pose_bad_cnt = 0
            else:
                st.pose_bad_cnt += 1

    def _toggle_auto_inspect(self):
        st = self.state
        st.auto_inspect = not st.auto_inspect
        st.last_auto_inspect_ts = 0.0
        st.status = "AUTO INSPECT ON" if st.auto_inspect else "AUTO INSPECT OFF"

    # -------------------------
    # Drawing
    # -------------------------
    def _draw_dev_hud(self, img):
        if not DEV_MODE:
            return
        h, w = img.shape[:2]
        if self.state.edit_mode:
            text1 = "EDIT: n=next  x=delete  r=clear  s=save  e=run"
        else:
            text1 = "RUN: SPACE=inspect  a=autoInspect  c=autotune  p=reload  e=edit"
        ovl = img.copy()
        overlay.draw_rect(ovl, (8, h-64), (w-8, h-8), color=(0,0,0), fill=True)
        cv2.addWeighted(ovl, 0.45, img, 0.55, 0, img)
        overlay.draw_text(img, text1, (16, h-80), color=(220,220,220), scale=0.6, thickness=1, align="lt")

        if not self.state.edit_mode:
            hint = "sample img [ T=temp  K:OK_S  N:NG_S ]"
            (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
            x = (w - tw)//2
            y = h - 22
            ovl2 = img.copy()
            cv2.addWeighted(ovl2, 0.45, img, 0.55, 0, img)
            overlay.draw_text(img, hint, (x, y), color=(220,220,220), scale=0.55, thickness=1, align="lt")

    def _draw_mode_indicator(self, img):
        h, w = img.shape[:2]
        text = "EDIT MODE" if self.state.edit_mode else "RUN MODE"
        color = (0,200,255) if self.state.edit_mode else (0,200,0)
        (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        x = w - tw - 16
        y = 28
        ovl = img.copy()
        overlay.draw_rect(ovl, (x-8, y-22), (x+tw+8, y+6), color=(0,0,0), fill=True)
        cv2.addWeighted(ovl, 0.4, img, 0.6, 0, img)
        overlay.draw_text(img, text, (x, y), color=color, scale=0.7, thickness=2, align="lt")
        if DEV_MODE:
            overlay.draw_text(img, "DEV", (x, y+20), color=(180,180,180), scale=0.5, thickness=1, align="lt")

    def _draw_pose_message(self, img):
        st = self.state
        # 조건: pose_bad_cnt >= N 이면 안내
        if st.pose_bad_cnt < POSE_BAD_N:
            return img

        msg = "정면으로 맞춰주세요 (±10~15°)"
        h, w = img.shape[:2]
        x = 30
        y = 80

        # bg
        ovl = img.copy()
        overlay.draw_rect(ovl, (x-12, y-30), (w-30, y+10), color=(0,0,0), fill=True)
        cv2.addWeighted(ovl, 0.45, img, 0.55, 0, img)

        # KR text (PIL)
        try:
            img = overlay.draw_text_kr(img, msg, (x, y-20))
        except Exception:
            # fallback english if font missing
            overlay.draw_text(img, "Align front (±10~15 deg)", (x, y-20), color=(255,255,255), scale=0.8, thickness=2, align="lt")
        return img

    # -------------------------
    # Sample capture (T/K/N)
    # -------------------------
    def _handle_sample_keys(self, key, frame_gray8, vis_bgr):
        st = self.state
        if st.edit_mode:
            return False

        if key not in (ord('t'), ord('T'), ord('k'), ord('K'), ord('n'), ord('N')):
            return False

        # selected ROI or ROI1
        try:
            sel = self.roi_mgr.get_selected()
        except Exception:
            sel = None
        roi_id = int(sel["id"]) if (isinstance(sel, dict) and sel.get("id") is not None) else 1

        is_t = key in (ord('t'), ord('T'))
        is_k = key in (ord('k'), ord('K'))
        is_n = key in (ord('n'), ord('N'))

        if is_t:
            crop = _crop_roi(frame_gray8, self.roi_mgr, roi_id)
            if crop is not None:
                tpath = os.path.join(DATA_DIR, "templates", f"tape_ok_ROI{roi_id}.png")
                cv2.imwrite(tpath, crop)
                print("[TEMPLATE SAVED]", tpath)
            return True

        tag = "OK" if is_k else "NG"
        out_dir = os.path.join(DATA_DIR, "dataset", tag)
        os.makedirs(out_dir, exist_ok=True)

        ts = time.strftime("%Y%m%d_%H%M%S")
        stem = f"cap_{ts}_ROI{roi_id}"

        raw_path = os.path.join(out_dir, f"{stem}_raw.png")
        ov_path  = os.path.join(out_dir, f"{stem}_overlay.png")
        crop_path = os.path.join(out_dir, f"{stem}_crop.png")

        cv2.imwrite(raw_path, frame_gray8)
        cv2.imwrite(ov_path, vis_bgr)

        crop = _crop_roi(frame_gray8, self.roi_mgr, roi_id)
        if crop is not None:
            cv2.imwrite(crop_path, crop)
        else:
            crop_path = ""

        meta = {"ts": ts, "tag": tag, "roi_id": roi_id, "raw": raw_path, "overlay": ov_path, "crop": crop_path}
        jpath = os.path.join(out_dir, f"{stem}.json")
        with open(jpath, "w", encoding="utf-8") as f:
            json.dump(meta, f, ensure_ascii=False, indent=2)

        prune_manifests(out_dir, keep=200)
        print("[SAVED]", meta)
        return True

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
                self._draw_run_tracking(vis, frame_gray8)

                # Auto inspect tick
                if st.auto_inspect:
                    now = time.time()
                    if (now - st.last_auto_inspect_ts) >= AUTO_INSPECT_INTERVAL:
                        st.last_auto_inspect_ts = now
                        self._inspect_once(frame_gray8, vis)

            # status + banner
            overlay.draw_status_bar(vis, st.status)
            if st.last_overall_ok is not None:
                overlay.draw_overall_banner(vis, st.last_overall_ok, info=_extract_info_from_results(st.last_results))

            # pose message
            vis = self._draw_pose_message(vis)

            # control bar (buttons)
            st.last_buttons = self._draw_control_bar(vis)

            # HUD
            self._draw_mode_indicator(vis)
            self._draw_dev_hud(vis)

            # show
            cv2.imshow(self.win, vis)

            # key
            key = cv2.waitKey(1) & 0xFF

            # sample keys first (consumes key)
            if self._handle_sample_keys(key, frame_gray8, vis):
                key = 255

            # keyboard -> pending cmd
            if key != 255:
                cmd = self._key_to_cmd(key)
                if cmd != UICmd.NONE:
                    st.pending_cmd = cmd

            # execute pending cmd
            if st.pending_cmd != UICmd.NONE:
                cmd_to_run = st.pending_cmd
                st.pending_cmd = UICmd.NONE
                self._execute(cmd_to_run, frame_gray8, vis)

            if st.quit_requested:
                break

        self.cam.release()
        cv2.destroyAllWindows()

    def _draw_control_bar(self, vis):
        st = self.state
        if st.edit_mode:
            control_buttons = [
                {"id":"toggle_edit","label":"RUN","color":(70,130,180),"enabled":True},
                {"id":"save","label":"SAVE","color":(120,0,120),"enabled":True},
                {"id":"next","label":"NEXT","color":(80,80,80),"enabled":True},
                {"id":"delete","label":"DELETE","color":(40,40,120),"enabled":True},
                {"id":"clear","label":"CLEAR","color":(40,0,80),"enabled":True},
                {"id":"quit","label":"QUIT","color":(120,0,0),"enabled":True},
            ]
        else:
            control_buttons = [
                {"id":"toggle_edit","label":"EDIT","color":(70,130,180),"enabled":True},
                {"id":"inspect","label":"INSPECT","color":(0,120,0),"enabled":True},
                {"id":"autoinspect","label":"AUTO","color":(80,80,80),"enabled":True},  # NEW
                {"id":"autotune","label":"AUTOTUNE","color":(0,120,120),"enabled":True},
                {"id":"reload","label":"RELOAD","color":(120,120,0),"enabled":True},
                {"id":"quit","label":"QUIT","color":(120,0,0),"enabled":True},
            ]
        return draw_control_bar(vis, control_buttons)

    def _execute(self, cmd: UICmd, frame_gray8, vis_bgr):
        st = self.state

        if cmd == UICmd.QUIT:
            st.quit_requested = True
            return

        if cmd == UICmd.TOGGLE_MODE:
            self._toggle_mode()
            return

        if cmd == UICmd.TOGGLE_AUTO_INSPECT:
            if st.edit_mode:
                st.status = "AUTO INSPECT only in RUN"
            else:
                self._toggle_auto_inspect()
            return

        if st.edit_mode:
            if cmd == UICmd.SAVE:
                self._save_roi_and_template(frame_gray8)
                return
            if cmd == UICmd.NEXT:
                try:
                    self.roi_mgr.select_next()
                    self.editor.on_select_changed()
                    st.status = f"Selected ROI: {self.roi_mgr.selected_id}"
                except Exception:
                    st.status = "Select next failed"
                return
            if cmd == UICmd.CLEAR:
                try:
                    if hasattr(self.roi_mgr, "clear"):
                        self.roi_mgr.clear()
                    else:
                        for r in list(self.roi_mgr.list()):
                            try: self.roi_mgr.remove(r["id"])
                            except Exception: pass
                    st.status = "Cleared ROIs"
                except Exception:
                    st.status = "Clear failed"
                return
            if cmd == UICmd.DELETE:
                try:
                    if hasattr(self.roi_mgr, "delete_selected"):
                        ok = self.roi_mgr.delete_selected()
                        st.status = "Deleted selected" if ok else "No ROI to delete"
                    else:
                        sid = self.roi_mgr.selected_id
                        if sid is not None:
                            self.roi_mgr.remove(sid)
                            st.status = f"Deleted ROI {sid}"
                        else:
                            st.status = "No ROI to delete"
                except Exception:
                    st.status = "Delete failed"
                return
            if cmd == UICmd.RELOAD:
                try:
                    if os.path.exists(ROI_PATH):
                        self.roi_mgr.load(ROI_PATH)
                        st.status = "ROI Reloaded"
                    else:
                        st.status = "No ROI file"
                except Exception as e:
                    st.status = f"Reload failed: {e}"
                return
            return

        # RUN mode
        if cmd == UICmd.INSPECT and not st.space_lock:
            try:
                st.space_lock = True
                self._inspect_once(frame_gray8, vis_bgr)
            finally:
                st.space_lock = False
            return

        if cmd == UICmd.AUTOTUNE:
            try:
                auto_path = os.path.join(ROI_DIR, "recipe_auto.json")
                self.inspector.autotune_recipe_from_frame(frame_gray8, save_path=auto_path)
                st.status = "Auto recipe saved"
            except Exception as e:
                st.status = f"Autotune failed: {e}"
            return

        if cmd == UICmd.RELOAD:
            try:
                self.inspector.reload_recipe()
                st.status = "Recipe reloaded"
            except Exception as e:
                st.status = f"Reload recipe failed: {e}"
            return

        if cmd == UICmd.SAVE:
            # keep your existing "commit moved rois" workflow out of this refactor (optional later)
            st.status = "Commit moved ROIs: not wired in refactor"
            return

    def _draw_run_tracking(self, vis, frame_gray8):
        """
        tracker + stabilizer overlay + snapshot keeping
        """
        st = self.state
        tracker = getattr(self.inspector, "tracker", None)

        moved = []
        try:
            if tracker is not None and getattr(tracker, "template", None) is not None and frame_gray8 is not None:
                if hasattr(self.roi_mgr, "list"):
                    rois_src = self.roi_mgr.list()
                elif hasattr(self.roi_mgr, "get_rois"):
                    rois_src = self.roi_mgr.get_rois()
                else:
                    rois_src = []

                for r in rois_src:
                    x = int(r.get("x", 0)); y = int(r.get("y", 0)); w = int(r.get("w", 0)); h = int(r.get("h", 0))
                    try:
                        out = tracker.track(frame_gray8, x, y, w, h) if hasattr(tracker, "track") else None
                        if out is None:
                            nx, ny, nw, nh = x, y, w, h
                        elif isinstance(out, (list, tuple)) and len(out) == 4:
                            nx, ny, nw, nh = map(int, out)
                        elif isinstance(out, (list, tuple)) and len(out) == 2:
                            dx, dy = map(int, out); nx, ny, nw, nh = x+dx, y+dy, w, h
                        elif isinstance(out, dict):
                            nx = int(out.get("x", x)); ny = int(out.get("y", y))
                            nw = int(out.get("w", w)); nh = int(out.get("h", h))
                        else:
                            nx, ny, nw, nh = x, y, w, h
                    except Exception:
                        nx, ny, nw, nh = x, y, w, h

                    moved.append({"id": r.get("id"), "name": r.get("name",""), "x": nx, "y": ny, "w": nw, "h": nh})

                smoothed, stable = self.stabilizer.update(moved)
                st.status = "RUN MODE (stable)" if stable else "RUN MODE (tracking...)"

                # snapshots when stable
                if stable and (time.time() - st.last_snapshot_time) > SNAPSHOT_COOLDOWN:
                    log_dir = os.path.join(DATA_DIR, "logs", "snapshots")
                    os.makedirs(log_dir, exist_ok=True)
                    for mr in smoothed:
                        roi_for_save = {"id": mr["id"], "x": int(round(mr["x"])), "y": int(round(mr["y"])), "w": int(mr["w"]), "h": int(mr["h"])}
                        save_snapshot(log_dir, frame_gray8, roi_for_save, prefix="stable")
                        prune_snapshots(log_dir, SNAPSHOT_KEEP)
                    if tracker is not None and getattr(tracker, "template", None) is not None:
                        save_template_copy(log_dir, tracker.template)
                    st.last_snapshot_time = time.time()

                # ROI overlay with results
                self._draw_roi_overlay(vis, moved, st.last_results)

            else:
                overlay.draw_rois(vis, rois=self._roi_mgr_to_list(), active_id=self.roi_mgr.selected_id, roi_results=st.last_results)

        except Exception as e:
            print("[DBG] run-mode tracker overlay exception:", e)
            overlay.draw_rois(vis, rois=self._roi_mgr_to_list(), active_id=self.roi_mgr.selected_id, roi_results=st.last_results)

    def _roi_mgr_to_list(self):
        return [
            {"id": r.get("id"), "label": r.get("name"),
             "rect": (int(r.get("x",0)), int(r.get("y",0)), int(r.get("w",0)), int(r.get("h",0)))}
            for r in getattr(self.roi_mgr, "rois", [])
        ]

    def _draw_roi_overlay(self, vis, moved, last_results):
        for mr in moved:
            rid_obj = mr.get("id")
            rid = str(rid_obj)

            lr = None
            if last_results:
                lr = last_results.get(rid)

            x = int(mr["x"]); y = int(mr["y"]); w = int(mr["w"]); h = int(mr["h"])

            if lr is None:
                overlay.draw_rect(vis, (x, y), (x + w, y + h), color=(0, 200, 0), thickness=2)
                line1 = f"ROI{rid}"
                tx, ty = roi_label_pos(x, y, w, h)
                overlay.draw_text(vis, line1, (tx, ty+14), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")
                continue

            metrics = getattr(lr, "metrics", None) or {}
            ok_flag = bool(getattr(lr, "ok", False))
            reason = getattr(lr, "reason", "") or ""

            box_color = (0, 200, 0) if ok_flag else (0, 0, 200)
            overlay.draw_rect(vis, (x, y), (x + w, y + h), color=box_color, thickness=2)

            line1 = f"ROI{rid} {'OK' if ok_flag else 'NG'}"
            if ok_flag:
                parts = []
                mean_v = metrics.get("mean", metrics.get("mean_raw", None))
                score_v = metrics.get("score", None)
                wr = metrics.get("white_ratio", None)
                edge = metrics.get("edge_energy", metrics.get("lap_var", metrics.get("laplacian_var", None)))
                qr = metrics.get("qr_data", None)
                bc = metrics.get("blob_count", None)

                if mean_v is not None: parts.append(f"m:{float(mean_v):.1f}")
                if score_v is not None: parts.append(f"s:{float(score_v):.2f}")
                if wr is not None: parts.append(f"wr:{float(wr):.2f}")
                if edge is not None: parts.append(f"e:{float(edge):.1f}")
                if bc is not None: parts.append(f"bc:{int(bc)}")
                if qr: parts.append(f"qr:{str(qr)[:12]}")
                line2 = " ".join(parts[:3]) if parts else ""
            else:
                line2 = str(reason)[:24] if reason else "FAIL"

            tx, ty = roi_label_pos(x, y, w, h)
            if line2:
                overlay.draw_text(vis, line2, (tx, ty), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")
            overlay.draw_text(vis, line1, (tx, ty+14), color=(255, 220, 20), scale=0.45, thickness=1, align="lt")


def main():
    app = VisionApp()
    app.run()

if __name__ == "__main__":
    main()