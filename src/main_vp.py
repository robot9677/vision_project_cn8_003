#!/usr/bin/env python3
import os
import time
import cv2
import numpy as np
import time
import json

from enum import Enum
from capture.camera_gst import CameraGST
from roi.roi_manager import ROIManager
from roi.roi_editor import ROIEditor
from inspection.inspector import Inspector
from ui.overlay import put_text_with_bg, draw_status_bar, draw_control_bar, draw_rois_clean
from ui import overlay_clean as overlay
from inspection.aligner import Aligner
from inspection.normalize import normalize_frame
from inspection.logger import save_snapshot, save_template_copy

# 개발자 모드: True 이면 화면에 키보드 도움말(개발자 HUD)을 표시
DEV_MODE = True

NORMALIZE_ENABLED = True
NORMALIZE_TARGET_MEAN = 120.0

# snapshot cooldown (초)
last_snapshot_time = 0.0
SNAPSHOT_COOLDOWN = 5.0

# =========================
# Command System
# =========================
class UICmd(Enum):
    NONE = 0
    TOGGLE_MODE = 1
    INSPECT = 2
    AUTO = 3
    RELOAD = 4
    SAVE = 5
    NEXT = 6
    CLEAR = 7
    QUIT = 8
    DELETE = 9

BUTTON_TO_CMD = {
    "toggle_edit": UICmd.TOGGLE_MODE,
    "inspect": UICmd.INSPECT,
    "autotune": UICmd.AUTO,
    "reload": UICmd.RELOAD,
    "save": UICmd.SAVE,
    "next": UICmd.NEXT,
    "nxt": UICmd.NEXT,       # backwards compat
    "clear": UICmd.CLEAR,
    "delete": UICmd.DELETE,
    "quit": UICmd.QUIT,
}

# =========================
# App Config
# =========================
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

# alignment template (파일 경로)
TEMPLATE_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "roi", "align_template.png")

def prune_snapshots(path, max_keep=200):
    try:
        files = []

        for fn in os.listdir(path):
            if fn.endswith(".png") or fn.endswith(".jpg"):
                p = os.path.join(path, fn)
                files.append((os.path.getmtime(p), p))

        files.sort(reverse=True)

        for _, p in files[max_keep:]:
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass

def roi_label_pos(x, y, w, h, margin=25):
    """
    ROI 라벨 위치 정책(한 군데만 수정)
    기본: ROI 박스 '위쪽' 좌측
    """
    tx = x
    ty = y - margin
    # 화면 위로 튀면 박스 아래로 내림
    if ty < 18:
        ty = y + h + 18
    return int(tx), int(ty)

def clahe_equalize(gray8, clip_limit=2.0, tile_grid_size=(8,8)):
    clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
    return clahe.apply(gray8)

def scale_to_target_mean(gray8, target_mean=120.0, max_scale=2.5, min_scale=0.5):
    m = float(max(1.0, gray8.mean()))
    scale = float(target_mean) / m
    scale = max(min_scale, min(max_scale, scale))
    # use convertScaleAbs for brightness scaling
    out = cv2.convertScaleAbs(gray8, alpha=scale, beta=0)
    return out, scale

def normalize_frame(gray8, target_mean=120.0, do_clahe=True):
    """
    Returns (normalized_gray8, info_dict)
    info_dict: {"scale":float, "method":"clahe|scale|both"}
    """
    if gray8 is None:
        return None, {"scale":1.0, "method":"none"}
    # quick clamp type
    if gray8.dtype != 'uint8':
        gray8 = cv2.normalize(gray8, None, 0, 255, cv2.NORM_MINMAX).astype('uint8')

    scaled, scale = scale_to_target_mean(gray8, target_mean=target_mean)
    method = "scale"
    if do_clahe:
        scaled = clahe_equalize(scaled)
        method = "scale+clahe"
    return scaled, {"scale": scale, "method": method}

def ensure_dirs():
    os.makedirs(ROI_DIR, exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "logs"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "images", "ok"), exist_ok=True)
    os.makedirs(os.path.join(DATA_DIR, "images", "ng"), exist_ok=True)

#  --- [start] sample image 수집 및 images 관리 ---
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
                # manifest에 기록된 파일들 삭제
                for k in ("raw", "overlay", "crop"):
                    p = meta.get(k)
                    if p and os.path.exists(p):
                        try: os.remove(p)
                        except Exception: pass
                os.remove(jpath)
            except Exception:
                # 실패해도 json은 지우기 시도
                try: os.remove(jpath)
                except Exception: pass
    except Exception:
        pass
#  --- [end] sample image 수집 및 images 관리 ---

def _extract_info_from_results(results):
    if not results:
        return {}

    total = len(results)
    ng = sum(1 for r in results.values() if not r.ok)

    info = {
        "total": total,
        "ng": ng,
    }

    # optional: 기존 정보 유지
    for r in results.values():
        m = r.metrics or {}
        if "norm_gain" in m:
            info["norm_gain"] = m["norm_gain"]
        if "dx" in m:
            info["dx"] = m["dx"]
        if "dy" in m:
            info["dy"] = m["dy"]
        break

    return info

def draw_dev_hud(img, edit_mode):
    """Draw keyboard help when DEV_MODE is True."""
    h, w = img.shape[:2]
    if edit_mode:
        text1 = "EDIT: n=next  x=delete  r=clear  s=save  e=run"
    else:
        text1 = "RUN: SPACE=inspect  c=auto  p=reload  e=edit"

    # semi-transparent background for readability
    ovl = img.copy()  # 이미지 복사 변수명은 ovl로 해서 모듈명 충돌 방지
    overlay.draw_rect(ovl, (8, h-64), (w-8, h-8), color=(0,0,0), fill=True)
    cv2.addWeighted(ovl, 0.45, img, 0.55, 0, img)

    # 텍스트는 overlay 모듈의 draw_text 사용
    overlay.draw_text(img, text1, (16, h-80), color=(220,220,220), scale=0.6, thickness=1, align="lt")

def draw_mode_indicator(img, edit_mode, dev_mode=False):
    """Draw mode indicator at top-right."""
    h, w = img.shape[:2]
    text = "EDIT MODE" if edit_mode else "RUN MODE"
    color = (0,200,255) if edit_mode else (0,200,0)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
    x = w - tw - 16
    y = 28

    # small background — use different var name so we don't shadow overlay module
    ovl = img.copy()
    # draw filled bg on ovl, then blend into img
    overlay.draw_rect(ovl, (x-8, y-22), (x+tw+8, y+6), color=(0,0,0), fill=True)
    cv2.addWeighted(ovl, 0.4, img, 0.6, 0, img)

    # draw main text via overlay module
    overlay.draw_text(img, text, (x, y), color=color, scale=None, thickness=None, align="lt")

    if dev_mode:
        dtxt = "DEV"
        overlay.draw_text(img, dtxt, (x, y+20), color=(180,180,180), scale=0.5, thickness=1, align="lt")

# --- overlay helper: convert ROIManager.rois to overlay format ---
def _roi_mgr_to_list(roi_mgr):
    if roi_mgr is None: return []
    return [ {"id": r.get("id"), "label": r.get("name"), "rect": (int(r.get("x",0)), int(r.get("y",0)), int(r.get("w",0)), int(r.get("h",0)))} for r in roi_mgr.rois ]

def main():
    ensure_dirs()

    cam = CameraGST(GST_PIPELINE)

    # --- CAMERA QUICK DIAG (임시) ---
    try:
        cam.open()
        for _i in range(5):
            frm = cam.read()
            if frm is None:
                print("[DBG CAMERA] read returned None")
                continue

            import cv2, numpy as np
            if len(frm.shape) == 2 or frm.shape[2] == 1:
                gray = frm.copy()   # 이미 mono
            else:
                gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)

            print(f"[DBG CAMERA] frame#{_i} mean:{gray.mean():.2f} min:{int(gray.min())} max:{int(gray.max())}")
            # 덤프는 한 번만 저장
            if _i == 0:
                try:
                    cv2.imwrite("/tmp/debug_frame_raw.jpg", frm)
                    print("[DBG CAMERA] dumped to /tmp/debug_frame_raw.jpg")
                except Exception as e:
                    print("[DBG CAMERA] dump failed:", e)
            # 잠깐 쉬어서 파이프라인 안정화 허용
            import time
            time.sleep(0.1)
        cam.release()
    except Exception as e:
        print("[DBG CAMERA] quick diag failed:", e)
    # --- end diag ---
    
    roi_mgr = ROIManager(frame_size=(WIDTH, HEIGHT))
    # load ROI file if exists; load returns bool but ignore failures
    try:
        roi_mgr.load(ROI_PATH)
        print("[DBG ROI COUNT]", len(getattr(roi_mgr, "rois", [])))
    except Exception:
        pass

    editor = ROIEditor(roi_mgr)
    inspector = Inspector(roi_mgr, recipe_path=RECIPE_PATH, logs_root=LOGS_ROOT)
    editor.on_select_changed = inspector.reset_tracker_template

    NORMALIZE_ENABLED = True
    NORMALIZE_TARGET_MEAN = 120.0

    from inspection.stabilizer import Stabilizer
    # ...
    stabilizer = Stabilizer(window=5, move_thresh_px=3, alpha=0.7)
    # auto-commit toggle (원하면 True)
    auto_commit_on_stable = False

    # --- TRACKER STARTUP DEBUG (insert after inspector init) ---
    try:
        if hasattr(inspector, "tracker"):
            tracker = inspector.tracker
            print("[DBG] inspector.tracker exists. attrs:", [a for a in dir(tracker) if not a.startswith("_")])
            # check if template is set (try common names)
            tpl_present = False
            if hasattr(tracker, "template") and getattr(tracker, "template") is not None:
                tpl_present = True
            if hasattr(tracker, "has_template") and callable(getattr(tracker, "has_template")):
                try:
                    tpl_present = tpl_present or bool(tracker.has_template())
                except Exception:
                    pass
            print(f"[DBG] tracker template present: {tpl_present}")
        else:
            print("[DBG] inspector.tracker NOT found on inspector")
    except Exception as e:
        print("[DBG] tracker startup debug exception:", e)
    # --- end startup debug ---

    # --- load saved alignment template if present ---
    try:
        if os.path.exists(TEMPLATE_PATH):
            tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
            if tpl is not None and hasattr(inspector, "tracker"):
                tracker = inspector.tracker
                if hasattr(tracker, "set_template"):
                    tracker.set_template(tpl)
                    print("[INFO] alignment template loaded into tracker.")
                else:
                    print("[WARN] inspector.tracker has no set_template()")
            else:
                print("[INFO] template file exists but tpl is None or inspector.tracker missing.")
    except Exception as e:
        print("[WARN] failed to load alignment template:", e)
    # --- end: load saved alignment template if present ---

    edit_mode = True
    status = "EDIT MODE"

    last_results = None
    last_overall_ok = None
    quit_requested = False
    space_lock = False

    # pending command set by mouse_router or keyboard; processed in main loop
    pending_cmd = None

    win = "Static Mode - ROI Setup"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, WIDTH, HEIGHT)

    last_ok_frame_time = time.time()
    last_buttons = []

    last_snapshot_time = 0.0
    SNAPSHOT_COOLDOWN = 5.0

    # =========================
    # Command Executor (runs in main loop)
    # =========================
    def execute_command(cmd, frame):
        nonlocal edit_mode, status, last_results, last_overall_ok, quit_requested, space_lock
        nonlocal pose_bad_cnt

        if cmd is None or cmd == UICmd.NONE:
            return

        # MODE toggle
        if cmd == UICmd.TOGGLE_MODE:
            edit_mode = not edit_mode
            try:
                inspector.mean_filter.reset()
            except Exception:
                pass
            try:
                if edit_mode:
                    # switched INTO EDIT mode -> clear in-memory template so user can set new one
                    if hasattr(inspector, "tracker") and hasattr(inspector.tracker, "set_template"):
                        inspector.tracker.set_template(None)
                else:
                    # switched INTO RUN mode -> if there is a saved template on disk, reload it into tracker
                    if os.path.exists(TEMPLATE_PATH):
                        tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
                        if tpl is not None and hasattr(inspector, "tracker") and hasattr(inspector.tracker, "set_template"):
                            inspector.tracker.set_template(tpl)
            except Exception:
                pass
            status = "EDIT MODE" if edit_mode else "RUN MODE"
            return

        # EDIT MODE commands
        if edit_mode:
            if cmd == UICmd.SAVE:
                try:
                    roi_mgr.save(ROI_PATH)

                    # frame : 현재 프레임 (BGR 또는 Gray)
                    frame_gray8 = None
                    try:
                        if frame is None:
                            print("[DEBUG] frame is None")
                        else:
                            # 이미 그레이일 경우 (height, width)
                            if len(frame.shape) == 2:
                                frame_gray8 = frame.copy()
                                print("[DEBUG] frame already gray, shape:", frame_gray8.shape)
                            # 컬러(BGR)일 경우 변환
                            elif len(frame.shape) == 3 and frame.shape[2] == 3:
                                frame_gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                                print("[DEBUG] frame converted BGR->GRAY, shape:", frame_gray8.shape)
                            else:
                                print("[DEBUG] frame has unexpected shape:", frame.shape)
                    except Exception as e:
                        print("[DEBUG] frame->gray exception:", e)
                        frame_gray8 = None

                    tpl_ok = False
                    try:
                        tpl_ok = roi_mgr.save_alignment_template(frame_gray8, TEMPLATE_PATH, roi_id=None)
                        print(f"[DEBUG] save_alignment_template returned: {tpl_ok}")
                    except Exception as e:
                        print("[DEBUG] save_alignment_template exception:", e)

                    # debug: 어떤 ROI를 쓰는지 출력
                    try:
                        sel = roi_mgr.get_selected()
                        print("[DEBUG] selected ROI:", sel)
                        all_rois = roi_mgr.get_rois()
                        print("[DEBUG] num rois:", len(all_rois))
                    except Exception as e:
                        print("[DEBUG] roi_mgr debug exception:", e)

                    # 파일 존재/크기 확인
                    if os.path.exists(TEMPLATE_PATH):
                        s = os.path.getsize(TEMPLATE_PATH)
                        print(f"[DEBUG] TEMPLATE_PATH exists: {TEMPLATE_PATH} size={s}")
                    else:
                        print(f"[DEBUG] TEMPLATE_PATH NOT FOUND: {TEMPLATE_PATH}")

                    status = "Saved ROI"
                except Exception as e:
                    status = f"Save failed: {e}"
                    print("[DEBUG] frame->gray 실패:", e)
                    return
                # 추가: 선택된 ROI 기준으로 alignment template도 저장 (그레이 이미지 필요)
                try:
                    frame_gray8 = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if (frame is not None and len(frame.shape) == 3) else (frame if frame is not None and len(frame.shape) == 2 else None)
                    ok_tpl = roi_mgr.save_alignment_template(frame_gray8, TEMPLATE_PATH, roi_id=None)
                    if ok_tpl and hasattr(inspector, "tracker"):
                        tpl = cv2.imread(TEMPLATE_PATH, cv2.IMREAD_GRAYSCALE)
                        if tpl is not None and hasattr(inspector.tracker, "set_template"):
                            inspector.tracker.set_template(tpl)
                            status = "Saved ROI + Template"

                except Exception:
                    # 실패해도 무시하되 상태는 ROI 저장으로 놔둠
                    pass
                return

            if cmd == UICmd.NEXT:
                try:
                    roi_mgr.select_next()
                    editor.on_select_changed()   # ← 이 1줄 추가 (또는 inspector.reset_tracker_template())
                    status = f"Selected ROI: {roi_mgr.selected_id}"
                except Exception:
                    status = "Select next failed"
                return
            if cmd == UICmd.CLEAR:
                try:
                    # prefer clear if exists, otherwise remove all via list
                    if hasattr(roi_mgr, "clear"):
                        roi_mgr.clear()
                    else:
                        for r in list(roi_mgr.list()):
                            try:
                                roi_mgr.remove(r["id"])
                            except Exception:
                                pass
                    status = "Cleared ROIs"
                except Exception:
                    status = "Clear failed"
                return
            if cmd == UICmd.DELETE:
                try:
                    if hasattr(roi_mgr, "delete_selected"):
                        ok = roi_mgr.delete_selected()
                        status = "Deleted selected" if ok else "No ROI to delete"
                    else:
                        sid = roi_mgr.selected_id
                        if sid is not None:
                            roi_mgr.remove(sid)
                            status = f"Deleted ROI {sid}"
                        else:
                            status = "No ROI to delete"
                except Exception:
                    status = "Delete failed"
                return
            if cmd == UICmd.RELOAD:
                try:
                    if os.path.exists(ROI_PATH):
                        roi_mgr.load(ROI_PATH)
                        print("[DBG ROI COUNT]", len(getattr(roi_mgr, "rois", [])))
                        status = "ROI Reloaded"
                    else:
                        status = "No ROI file"
                except Exception as e:
                    status = f"Reload failed: {e}"
                return

        # RUN MODE commands
        else:
            # Allow SAVE while in RUN to commit tracked/moved ROIs to roi.json
            if cmd == UICmd.SAVE and not edit_mode:
                try:
                    # require a tracker/template and a recent frame
                    tracker = getattr(inspector, "tracker", None)
                    if tracker is None or getattr(tracker, "template", None) is None:
                        status = "No tracker/template to commit"
                        return

                    # Build moved list exactly the same way RUN drawing does
                    frame_gray8 = frame if (hasattr(frame, "ndim") and frame.ndim == 2) else (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None)
                    if frame_gray8 is None:
                        status = "No frame to commit"
                        return

                    if hasattr(roi_mgr, "list"):
                        rois_src = roi_mgr.list()
                    elif hasattr(roi_mgr, "get_rois"):
                        rois_src = roi_mgr.get_rois()
                    else:
                        rois_src = []

                    moved_rois = []
                    for r in rois_src:
                        x = int(r.get("x",0)); y = int(r.get("y",0)); w = int(r.get("w",0)); h = int(r.get("h",0))
                        try:
                            out = tracker.track(frame_gray8, x, y, w, h) if hasattr(tracker, "track") else None
                            if out is None:
                                nx, ny, nw, nh = x, y, w, h
                            elif isinstance(out, (list,tuple)) and len(out) == 4:
                                nx, ny, nw, nh = map(int, out)
                            elif isinstance(out, (list,tuple)) and len(out) == 2:
                                dx, dy = map(int, out); nx, ny, nw, nh = x+dx, y+dy, w, h
                            elif isinstance(out, dict):
                                nx = int(out.get("x", x)); ny = int(out.get("y", y))
                                nw = int(out.get("w", w)); nh = int(out.get("h", h))
                            else:
                                nx, ny, nw, nh = x, y, w, h
                        except Exception:
                            nx, ny, nw, nh = x, y, w, h
                        # preserve id/name when possible
                        moved_rois.append({"id": r.get("id"), "name": r.get("name",""), "x": int(nx), "y": int(ny), "w": int(nw), "h": int(nh)})

                    # Try to write back into roi_mgr internal structure safely
                    written = False
                    try:
                        # best-effort: set attribute names commonly used
                        if hasattr(roi_mgr, "rois"):
                            roi_mgr.rois = moved_rois
                            written = True
                        elif hasattr(roi_mgr, "_rois"):
                            roi_mgr._rois = moved_rois
                            written = True
                        elif hasattr(roi_mgr, "replace_all"):
                            roi_mgr.replace_all(moved_rois)
                            written = True
                        elif hasattr(roi_mgr, "save"):  # fallback: overwrite by saving file directly
                            # overwrite ROI file with moved_rois (as list)
                            with open(ROI_PATH, "w") as f:
                                json.dump({"rois": moved_rois}, f, indent=2)
                            written = True
                    except Exception:
                        written = False

                    if written:
                        status = "Committed moved ROIs"
                    else:
                        status = "Commit failed (no write method)"

                except Exception as e:
                    status = f"Commit failed: {e}"
                return
            
            if cmd == UICmd.INSPECT and not space_lock:
                try:
                    space_lock = True

                    # 0) tracker 확보 (로컬 스코프)
                    tracker = getattr(inspector, "tracker", None)

                    # 1) grayscale frame
                    frame_gray = None
                    try:
                        if frame is None:
                            frame_gray = None
                        elif hasattr(frame, "ndim") and frame.ndim == 2:
                            frame_gray = frame
                        elif hasattr(frame, "shape") and frame.shape[2] == 3:
                            frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
                        else:
                            frame_gray = frame
                    except Exception:
                        frame_gray = frame

                    # 2) 5프레임 평균
                    frames = []
                    for _ in range(5):
                        f = cam.read()
                        if f is None:
                            continue
                        if getattr(f, "ndim", 0) == 3:
                            f = f[:, :, 0]
                        frames.append(f)

                    avg = np.mean(frames, axis=0).astype("uint8") if frames else frame_gray

                    # 3) (선택) INSPECT 직전 1회 더 트래킹 보정: moved가 있을 때만
                    # moved 변수는 RUN loop에서 계산된 걸 쓰는 구조라면 여기서 존재할 수 있음
                    if tracker is not None and avg is not None and "moved" in locals():
                        moved2 = []
                        for r in moved:
                            x = int(r["x"]); y = int(r["y"]); w = int(r["w"]); h = int(r["h"])
                            try:
                                nx, ny, nw, nh = tracker.track(avg, x, y, w, h)
                            except Exception:
                                nx, ny, nw, nh = x, y, w, h
                            moved2.append({"id": r.get("id"), "name": r.get("name",""), "x": nx, "y": ny, "w": nw, "h": nh})
                        moved = moved2

                    # 4) inspect 딱 1번
                    overall_ok, results = inspector.inspect(avg)

                    # 5) save/log/ui
                    try:
                        run_dir = inspector.save_run(avg, vis.copy(), overall_ok, results)
                    except Exception as e:
                        print("[DBG] save_run failed:", e)

                    last_results = {str(k): v for k, v in results.items()} if results else {}
                    last_overall_ok = overall_ok
                    status = f"INSPECT {'OK' if overall_ok else 'NG'}"
                    inspector.log_result(last_overall_ok, last_results)

                    # ------[START] ROI내에서 제품 정면 아닌 각도가 틀어진 경우 안내문구 표시를 위한 작업
                    r = (last_results or {}).get("1") or (last_results or {}).get("ROI1")
                    bc = None
                    if isinstance(r, dict):
                        bc = (r.get("metrics") or {}).get("blob_count", None)

                    if bc == 4:
                        pose_bad_cnt = 0
                    else:
                        pose_bad_cnt += 1
                    # ------[END] ROI내에서 제품 정면 아닌 각도가 틀어진 경우 안내문구 표시를 위한 작업

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    status = f"Inspect failed: {e}"
                finally:
                    space_lock = False
                return

            if cmd == UICmd.AUTO:
                try:
                    auto_path = os.path.join(ROI_DIR, "recipe_auto.json")
                    inspector.autotune_recipe_from_frame(frame, save_path=auto_path)
                    status = "Auto recipe saved"
                    print("[DBG] auto recipe saved:")
                except Exception as e:
                    status = f"Autotune failed: {e}"
                    print("[DBG] Autotune failed:", e)
                return

            if cmd == UICmd.RELOAD:
                try:
                    inspector.reload_recipe()
                    status = "Recipe reloaded"
                except Exception as e:
                    status = f"Reload recipe failed: {e}"
                return

        if cmd == UICmd.QUIT:
            quit_requested = True
            return

    # =========================
    # Mouse Router: SETS pending_cmd, doesn't execute
    # =========================
    def mouse_router(event, x, y, flags, param):
        nonlocal last_buttons, edit_mode, pending_cmd

        # 1) MOVE: 편집모드일 때만 editor에 전달
        if event == cv2.EVENT_MOUSEMOVE:
            if edit_mode:
                editor._on_mouse(event, x, y, flags, None)
            return

        # 2) LEFT DOWN만 버튼 처리 + 편집 전달
        if event == cv2.EVENT_LBUTTONDOWN:
            # 버튼 hit
            for b in last_buttons:
                x1, y1, x2, y2 = b.get("rect", (0,0,0,0))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    if b.get("enabled", True):
                        bid = b.get("id")
                        pending_cmd = BUTTON_TO_CMD.get(bid, UICmd.NONE)
                    return

            # 버튼 아니면 편집모드일 때만 editor
            if edit_mode:
                editor._on_mouse(event, x, y, flags, None)
            return
        
        # 3) 나머지 이벤트(UP, RBUTTON 등): 편집모드일 때만 editor
        if edit_mode:
            editor._on_mouse(event, x, y, flags, None)

    cv2.setMouseCallback(win, mouse_router)

    cam.open()

    pose_bad_cnt = 0
    POSE_BAD_N = 5   # 연속 5프레임이면 안내 표시

    # main loop
    while True:

        frame = cam.read()
        if frame is None:
            if time.time() - last_ok_frame_time > 1.0:
                status = "No frames >1s (check camera)"
            cv2.waitKey(1)
            if quit_requested:
                break
            continue

        last_ok_frame_time = time.time()

        # ensure GRAY8 single channel
        if frame.ndim == 3:
            frame = frame[:, :, 0]

        vis = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)

        # Overlays depending on mode
        if edit_mode:
            # editor draws handles/selection on vis
            editor.update(vis)
           # overlay.draw_rois(vis, rois=_roi_mgr_to_list(roi_mgr), active_id=roi_mgr.selected_id)
        else:
            # RUN mode: single, deterministic tracker pass -> draw moved ROIs
            info_save = _extract_info_from_results(last_results)

            # prepare gray frame (frame already GRAY8)
            frame_gray8 = frame if (hasattr(frame, "ndim") and frame.ndim == 2) else (cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY) if frame is not None else None)

            # 정규화 적용
            if NORMALIZE_ENABLED and frame_gray8 is not None:
                frame_gray8, ninfo = normalize_frame(frame_gray8, target_mean=NORMALIZE_TARGET_MEAN, do_clahe=True)
                # status 짧게 표시에 쓰려면 status = f"norm:{ninfo['method']} s={ninfo['scale']:.2f}"

            tracker = getattr(inspector, "tracker", None)
            moved = []

            try:
                # if tracker and template present -> track each ROI once
                if tracker is not None and getattr(tracker, "template", None) is not None and frame_gray8 is not None:
                    # get list of rois in a compatible way
                    if hasattr(roi_mgr, "list"):
                        rois_src = roi_mgr.list()
                    elif hasattr(roi_mgr, "get_rois"):
                        rois_src = roi_mgr.get_rois()
                    else:
                        rois_src = []

                    # Single pass: call tracker.track(...) per ROI and collect results
                    for r in rois_src:
                        x = int(r.get("x", 0)); y = int(r.get("y", 0)); w = int(r.get("w", 0)); h = int(r.get("h", 0))
                        try:
                            # Expecting (nx,ny,nw,nh) per debug logs
                            out = None
                            if hasattr(tracker, "track"):
                                out = tracker.track(frame_gray8, x, y, w, h)
                            elif hasattr(tracker, "update"):
                                out = tracker.update(frame_gray8, x, y, w, h)
                            # normalize output
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
                        except Exception as e:
                            # fallback to original ROI on error
                            nx, ny, nw, nh = x, y, w, h
                        moved.append({"id": r.get("id"), "name": r.get("name",""), "x": nx, "y": ny, "w": nw, "h": nh})

                        # moved: list of dicts with integer coords from tracker
                        # apply stabilizer -> smoothed (float coords) + stable boolean
                        smoothed, stable = stabilizer.update(moved)

                        # draw using smoothed coords (round to int when drawing)
                        for mr in smoothed:
                            x = int(round(mr["x"])); y = int(round(mr["y"]))
                            w = int(mr["w"]); h = int(mr["h"])
                            # overlay.draw_rect(vis, (x, y), (x + w, y + h), color=(0, 200, 0), thickness=2)
                            # if mr.get("name"):
                            #     cv2.putText(vis, str(mr["name"]), (x, max(12, y-6)), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0,200,0), 1, cv2.LINE_AA)

                        # optionally show stability on status bar / overlay
                        if stable:
                            status = "RUN MODE (stable)"
                            if stable and (time.time() - last_snapshot_time) > SNAPSHOT_COOLDOWN:
                                log_dir = os.path.join(DATA_DIR, "logs", "snapshots")
                                # smoothed: list of smoothed roi dicts (int coords recommended)
                                for mr in smoothed:
                                    roi_for_save = {"id": mr["id"], "x": int(round(mr["x"])), "y": int(round(mr["y"])), "w": int(mr["w"]), "h": int(mr["h"])}
                                    save_snapshot(log_dir, frame_gray8, roi_for_save, prefix="stable")
                                    prune_snapshots(log_dir)
                                if hasattr(inspector, "tracker") and getattr(inspector.tracker, "template", None) is not None:
                                    save_template_copy(log_dir, inspector.tracker.template)
                                last_snapshot_time = time.time()
                        else:
                            status = "RUN MODE (tracking...)"

                        # auto commit if enabled and stable
                        if auto_commit_on_stable and stable:
                            # call the same commit routine we discussed earlier (best-effort write)
                            # you can reuse the commit block from earlier: build moved_rois from smoothed (round ints), then write back.
                            try:
                                # build int list
                                commit_rois = []
                                for mr in smoothed:
                                    commit_rois.append({"id": mr.get("id"), "name": mr.get("name",""), "x": int(round(mr["x"])), "y": int(round(mr["y"])), "w": int(mr["w"]), "h": int(mr["h"])})
                                # write back (best-effort)
                                with open(ROI_PATH, "w") as f:
                                    json.dump({"rois": commit_rois}, f, indent=2)
                                status = "Committed moved ROIs (auto)"
                            except Exception as e:
                                print("[STAB] auto-commit failed:", e)

                    # --- ROI overlay (single source of truth) ---
                    try:
                        for mr in moved:
                            rid_obj = mr.get("id")
                            rid = str(rid_obj)

                            lr = None
                            if last_results:
                                lr = last_results.get(rid) or (last_results.get(int(rid_obj)) if rid_obj is not None else None)

                            x = int(mr["x"]); y = int(mr["y"]); w = int(mr["w"]); h = int(mr["h"])

                            # (1) default: just green box (no text) when no result
                            if lr is None:
                                overlay.draw_rect(vis, (x, y), (x + w, y + h), color=(0, 200, 0), thickness=2)

                                # 라벨 1줄(ROI 번호만) 항상 표시
                                line1 = f"ROI{rid}"
                                tx, ty = roi_label_pos(x, y, w, h)
                                overlay.draw_text(vis, line1, (tx, ty+14), color=(255, 220, 20), scale=0.45, thickness=1)
                                continue

                            # (2) extract result
                            if hasattr(lr, "metrics"):
                                metrics = lr.metrics or {}
                                ok_flag = bool(lr.ok)
                                reason = getattr(lr, "reason", "") or ""
                            elif isinstance(lr, dict):
                                metrics = (lr.get("metrics") or {})
                                ok_flag = bool(lr.get("ok", False))
                                reason = lr.get("reason", "") or ""
                            else:
                                metrics = {}
                                ok_flag = False
                                reason = ""

                            # (3) box color
                            box_color = (0, 200, 0) if ok_flag else (0, 0, 200)
                            overlay.draw_rect(vis, (x, y), (x + w, y + h), color=box_color, thickness=2)

                            # (4) 2-line label
                            line1 = f"ROI{rid} {'OK' if ok_flag else 'NG'}"
                            if ok_flag:
                                mean_v = metrics.get("mean", metrics.get("mean_raw", None))
                                score_v = metrics.get("score", None)
                                wr = metrics.get("white_ratio", None)
                                edge = metrics.get("edge_energy", metrics.get("lap_var", metrics.get("laplacian_var", None)))
                                qr = metrics.get("qr_data", None)

                                parts = []
                                if mean_v is not None:
                                    parts.append(f"m:{float(mean_v):.1f}")
                                if score_v is not None:
                                    parts.append(f"s:{float(score_v):.2f}")
                                if wr is not None:
                                    parts.append(f"wr:{float(wr):.2f}")
                                if edge is not None:
                                    parts.append(f"e:{float(edge):.1f}")
                                if qr:
                                    parts.append(f"qr:{str(qr)[:12]}")

                                bc = metrics.get("blob_count", None)
                                if bc is not None:
                                    parts.append(f"bc:{int(bc)}")

                                line2 = " ".join(parts[:3]) if parts else ""
                            else:
                                line2 = str(reason)[:24] if reason else "FAIL"

                            tx, ty = roi_label_pos(x, y, w, h)
                            
                            if line2:
                                overlay.draw_text(vis, line2, (tx, ty), color=(255, 220, 20), scale=0.45, thickness=1)
                            overlay.draw_text(vis, line1, (tx, ty+14), color=(255, 220, 20), scale=0.45, thickness=1)

                    except Exception:
                        pass
                    # --- end overlay ---
                    
                else:
                    # no valid tracker/template -> draw original ROI widgets
                    overlay.draw_rois(vis, rois=_roi_mgr_to_list(roi_mgr), active_id=roi_mgr.selected_id, roi_results=last_results)

            except Exception as e:
                # safe fallback: draw original roi UI and log
                print("[DBG] run-mode tracker overlay exception:", e)
                overlay.draw_rois(vis, rois=_roi_mgr_to_list(roi_mgr), active_id=roi_mgr.selected_id, roi_results=last_results)

        # status line
        overlay.draw_status_bar(vis, status)

        if last_overall_ok is not None:
            overlay.draw_overall_banner(vis, last_overall_ok, info=_extract_info_from_results(last_results))

        # --- pose bc read (always from last_results) ---
        bc_pose = None
        r = (last_results or {}).get("1")  # ROI1 id가 문자열 "1"로 저장됨
        if r is not None and hasattr(r, "metrics"):
            bc_pose = (r.metrics or {}).get("blob_count", None)
        elif isinstance(r, dict):
            bc_pose = (r.get("metrics") or {}).get("blob_count", None)

        show_pose_msg = (bc_pose is not None) and (pose_bad_cnt >= POSE_BAD_N) and (int(bc_pose) != 4)

        # --- pose assist message ---
        if show_pose_msg:
            h, w = vis.shape[:2]
            msg = "정면으로 맞춰주세요 (±10~15°)"
            (tw, th), _ = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
            x = (w - tw)//2
            y = 80   # 상단 중앙

            ovl = vis.copy()
            overlay.draw_rect(ovl, (x-14, y-26), (x+tw+14, y+10), color=(0,0,0), fill=True)
            cv2.addWeighted(ovl, 0.45, vis, 0.55, 0, vis)

            vis = overlay.draw_text_kr(vis, msg, (x, y))

        # --- keyboard fallback: set pending_cmd, don't execute directly ---
        key = cv2.waitKey(1) & 0xFF

        # ---[START] RUN mode sample capture (T/K/N) ---
        if (not edit_mode) and key in (ord('t'), ord('T'), ord('k'), ord('K'), ord('n'), ord('N')):
            # 선택 ROI 없으면 ROI1로
            sel = None
            try:
                sel = roi_mgr.get_selected()
            except Exception:
                sel = None
            roi_id = int(sel["id"]) if (isinstance(sel, dict) and sel.get("id") is not None) else 1

            # 저장 폴더
            base = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "dataset"))
            is_t = key in (ord('t'), ord('T'))
            is_k = key in (ord('k'), ord('K'))
            is_n = key in (ord('n'), ord('N'))

            tag = "OK" if is_k else ("NG" if is_n else "TEMPLATE")

            # T는 dataset에 저장 안 함
            if not is_t:
                out_dir = os.path.join(base, tag)
                os.makedirs(out_dir, exist_ok=True)

            if is_t:
                # 템플릿 1장 덮어쓰기
                tdir = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "data", "templates"))
                os.makedirs(tdir, exist_ok=True)
                crop = _crop_roi(frame_gray8, roi_mgr, roi_id)
                if crop is not None:
                    tpath = os.path.join(tdir, f"tape_ok_ROI{roi_id}.png")
                    cv2.imwrite(tpath, crop)
                    print("[TEMPLATE SAVED]", tpath)
            else:
                ts = time.strftime("%Y%m%d_%H%M%S")
                stem = f"cap_{ts}_ROI{roi_id}"

                # raw(그레이) / overlay(BGR) / crop 저장
                raw_path = os.path.join(out_dir, f"{stem}_raw.png")
                ov_path  = os.path.join(out_dir, f"{stem}_overlay.png")
                crop_path = os.path.join(out_dir, f"{stem}_crop.png")

                cv2.imwrite(raw_path, frame_gray8)
                cv2.imwrite(ov_path, vis)

                crop = _crop_roi(frame_gray8, roi_mgr, roi_id)
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
                key = 255  # 아래 keymap 처리가 이 키를 다시 먹지 않게
        # ---[END] RUN mode sample capture (T/K/N) ---


        if key != 255:
            keymap = {
                27: UICmd.QUIT, ord('q'): UICmd.QUIT,
                ord('e'): UICmd.TOGGLE_MODE,
                ord('s'): UICmd.SAVE,
                ord('n'): UICmd.NEXT,
                ord('r'): UICmd.CLEAR,
                ord('p'): UICmd.RELOAD,
                ord('c'): UICmd.AUTO,
                32: UICmd.INSPECT,
                ord('x'): UICmd.DELETE,
            }
            pending_cmd = keymap.get(key, pending_cmd)

        # --- Execute pending command BEFORE drawing control bar so UI updates immediately ---
        if pending_cmd is not None and pending_cmd != UICmd.NONE:
            cmd_to_run = pending_cmd
            pending_cmd = None
            execute_command(cmd_to_run, frame)

        # --- Build control buttons based on current mode and draw them (last so they reflect state) ---
        if edit_mode:
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
                {"id":"autotune","label":"AUTO","color":(0,120,120),"enabled":True},
                {"id":"reload","label":"RELOAD","color":(120,120,0),"enabled":True},
                {"id":"quit","label":"QUIT","color":(120,0,0),"enabled":True},
            ]

        # draw control bar and remember button rects for hit testing
        last_buttons = draw_control_bar(vis, control_buttons)

        # draw mode indicator + dev HUD if enabled
        draw_mode_indicator(vis, edit_mode, DEV_MODE)
        if DEV_MODE:
            draw_dev_hud(vis, edit_mode)
            # RUN mode key hint (bottom-center)
            if not edit_mode:
                h, w = vis.shape[:2]
                hint = "sample img [ T=temp  K:OK_S  N:NG_S ]"
                (tw, th), _ = cv2.getTextSize(hint, cv2.FONT_HERSHEY_SIMPLEX, 0.55, 1)
                x = (w - tw)//2
                y = h - 22
                ovl = vis.copy()
                # overlay.draw_rect(ovl, (x-5, y-5), (x+tw+10, y+8), color=(0,0,0), fill=True)
                cv2.addWeighted(ovl, 0.45, vis, 0.55, 0, vis)
                overlay.draw_text(vis, hint, (x+80, y-60), color=(220,220,220), scale=0.55, thickness=1, align="lt")

        cv2.imshow(win, vis)

        if quit_requested:
            break

    cam.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()