#!/usr/bin/env python3
import sys
import os
import json
import cv2
import numpy as np

# ensure src is importable when running from tools/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))
from roi.roi_manager import ROIManager
from roi.roi_editor import ROIEditor
from ui import overlay_clean as overlay


# ===== 판정 임계값 (그레이 평균 기준, 단위: gray level) =====
BASELINE_FILE = "baseline.json"   # baseline 저장 파일 (프로젝트 루트)
OK_DIFF = 8        # baseline 대비 ±8 이내 -> OK (green)
WARN_DIFF = 18     # ±18 이내 -> WARN (yellow)
# 그 이상 -> NG (red)


def main():
    src = sys.argv[1] if len(sys.argv) > 1 else "0"
    # if src is digit -> camera index
    cap = None
    is_video = False
    if str(src).isdigit():
        src_idx = int(src)
        cap = cv2.VideoCapture(src_idx)
        is_video = True
    elif os.path.exists(src):
        # image or video file
        # try opening as image first
        img = cv2.imread(src)
        if img is None:
            cap = cv2.VideoCapture(src)
            is_video = True
        else:
            cap = None
            frame = img
    else:
        print("Source not found:", src)
        return

    if cap is not None and not cap.isOpened():
        print("Failed to open capture:", src)
        return

    # Determine frame size (grab first frame if needed)
    if cap is not None:
        ret, frame = cap.read()
        if not ret:
            print("Failed reading first frame")
            return
    h, w = frame.shape[:2]

    roi_mgr = ROIManager((w, h))
    roi_editor = ROIEditor(roi_mgr, min_size=10)

    win = "ROI Inspector"
    cv2.namedWindow(win, cv2.WINDOW_NORMAL)
    cv2.resizeWindow(win, 1280, 720)
    roi_editor.attach_window(win)

    save_path = "roi_saved.json"

    # state variables
    show_stats = False
    baseline = {}  # ROI id -> baseline gray mean

    # baseline persistence helpers
    def save_baseline(path=BASELINE_FILE):
        try:
            with open(path, 'w', encoding='utf-8') as f:
                json.dump(baseline, f, indent=2)
            print(f"Baseline saved to {path}")
        except Exception as e:
            print("Failed to save baseline:", e)

    def load_baseline(path=BASELINE_FILE):
        try:
            if not os.path.exists(path):
                print("Baseline file not found:", path)
                return
            with open(path, 'r', encoding='utf-8') as f:
                data = json.load(f)
            # ensure numeric conversion
            for k,v in data.items():
                baseline[int(k)] = float(v)
            print(f"Baseline loaded from {path}")
        except Exception as e:
            print("Failed to load baseline:", e)

    print("Controls: t = toggle stats, b = set baseline (selected ROI or all), r = reset baselines, s = save ROIs, l = load ROIs, n = rename, q/Esc = quit")

    # auto-load baseline if present
    if os.path.exists(BASELINE_FILE):
        load_baseline(BASELINE_FILE)

    while True:
        if cap is not None:
            ret, frame = cap.read()
            if not ret:
                # video end -> break
                break

        # make a working copy for display
        display = frame.copy()

        # ensure display is 3-channel BGR
        if display.ndim == 2 or (display.ndim == 3 and display.shape[2] == 1):
            display = cv2.cvtColor(display, cv2.COLOR_GRAY2BGR)

        # draw ROIs (editor draws boxes/handles)
        roi_editor.update(display)

        # compute & optionally draw statistics / states for each ROI
        if True:
            # We'll compute state for each ROI and draw state-color rim even if show_stats is off,
            # but textual stats only when show_stats == True.
            for idx, r in enumerate(roi_mgr.list()):
                x, y, w_, h_ = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
                # clamp coordinates
                x1 = max(0, min(x, frame.shape[1] - 1))
                y1 = max(0, min(y, frame.shape[0] - 1))
                x2 = max(0, min(x + w_, frame.shape[1]))
                y2 = max(0, min(y + h_, frame.shape[0]))

                if x2 <= x1 or y2 <= y1:
                    continue

                roi_img = frame[y1:y2, x1:x2]
                if roi_img.size == 0:
                    continue

                # ensure BGR
                if roi_img.ndim == 2:
                    roi_bgr = cv2.cvtColor(roi_img, cv2.COLOR_GRAY2BGR)
                else:
                    roi_bgr = roi_img

                # grayscale mean/std
                gray = cv2.cvtColor(roi_bgr, cv2.COLOR_BGR2GRAY)
                mean_gray = float(np.mean(gray))
                std_gray = float(np.std(gray))

                # determine baseline/state
                rid = r["id"]
                state = "UNKNOWN"
                color = (200, 200, 200)  # default gray for unknown
                if rid in baseline:
                    diff = abs(mean_gray - baseline[rid])
                    if diff <= OK_DIFF:
                        state = "OK"
                        color = (0, 200, 0)    # green
                    elif diff <= WARN_DIFF:
                        state = "WARN"
                        color = (0, 200, 200)  # yellow-ish (BGR)
                    else:
                        state = "NG"
                        color = (0, 0, 200)    # red-ish (BGR)

                # draw colored rim to indicate state (always visible)
                if state == "UNKNOWN":
                    # light blue for unknown
                    rim_color = (200, 200, 255)
                    rim_th = 2
                else:
                    rim_color = color
                    rim_th = 3 if state != "UNKNOWN" else 2

                # Draw rim: outer white for selected ROI, else colored rectangle
                if r["id"] == roi_mgr.selected_id:
                    overlay.draw_rect(display, (x1, y1), (x2, y2), color=(255, 255, 255), thickness=4)
                    cv2.rectangle(display, (x1+2, y1+2), (x2-2, y2-2), rim_color, 2, lineType=cv2.LINE_AA)
                else:
                    cv2.rectangle(display, (x1, y1), (x2, y2), rim_color, rim_th, lineType=cv2.LINE_AA)

                # if stats enabled, show numeric info and a small swatch
                if show_stats:
                    # channel means
                    ch_means = cv2.mean(roi_bgr)[:3]
                    ch_str = f"B:{ch_means[0]:.0f} G:{ch_means[1]:.0f} R:{ch_means[2]:.0f}"
                    gs_str = f"{state} g_mean:{mean_gray:.0f} g_std:{std_gray:.0f}"

                    # ----- TEXT POSITION FIX (항상 ROI 밖에 표시) -----
                    text_margin = 55
                    line_h = 20   # 글자 1줄 높이 (font 0.9 기준)

                    tx = x1

                    # 위쪽에 표시 시도
                    ty = y1 - text_margin

                    # 위 공간 부족하면 아래쪽으로 이동
                    if ty - line_h < 0:
                        ty = y2 + text_margin

                    # draw shadow + text
                    overlay.draw_text(display, ch_str, (tx, ty), color=(0, 0, 0), scale=0.7, thickness=4)
                    overlay.draw_text(display, ch_str, (tx, ty), color=(255, 255, 255), scale=0.7, thickness=2)

                    overlay.draw_text(display, gs_str, (tx, ty + line_h), color=(0, 0, 0), scale=0.7, thickness=4)
                    overlay.draw_text(display, gs_str, (tx, ty + line_h), color=(255, 255, 255), scale=0.7, thickness=2)

                    # small color swatch (average color)
                    sw = 22
                    sw_x = min(display.shape[1] - sw - 6, x1 + w_ + 6)
                    sw_y = y1
                    avg_color = (int(ch_means[0]), int(ch_means[1]), int(ch_means[2]))
                    cv2.rectangle(display, (sw_x, sw_y), (sw_x + sw, sw_y + sw), avg_color, -1, lineType=cv2.LINE_AA)
                    overlay.draw_rect(display, (sw_x, sw_y), (sw_x + sw, sw_y + sw), color=(0, 0, 0), thickness=1)

        cv2.imshow(win, display)
        k = cv2.waitKey(30) & 0xFF
        if k == ord('q') or k == 27:
            break
        elif k == ord('s'):
            roi_mgr.save(save_path)
            print("Saved ROIs to", save_path)
        elif k == ord('l'):
            if os.path.exists(save_path):
                roi_mgr.load(save_path)
                print("Loaded ROIs from", save_path)
            else:
                print("No saved ROI file:", save_path)
        elif k == ord('n'):
            # rename selected via terminal input
            sid = roi_mgr.selected_id
            if sid is None:
                print("No ROI selected to rename.")
            else:
                try:
                    newname = input(f"Enter new name for ROI#{sid}: ").strip()
                    if newname:
                        roi_mgr.update(sid, name=newname)
                        print("Renamed.")
                except Exception as e:
                    print("Rename failed:", e)
        elif k == ord('t'):
            # toggle statistics display
            show_stats = not show_stats
            print("ROI stats display:", "ON" if show_stats else "OFF")
        elif k == ord('b'):
            # learn baseline: if ROI selected -> only that, else all ROIs
            sid = roi_mgr.selected_id
            if sid is None:
                # set baseline for all ROIs using current frame values
                updated = 0
                for r in roi_mgr.list():
                    x, y, w_, h_ = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
                    x1 = max(0, min(x, frame.shape[1] - 1))
                    y1 = max(0, min(y, frame.shape[0] - 1))
                    x2 = max(0, min(x + w_, frame.shape[1]))
                    y2 = max(0, min(y + h_, frame.shape[0]))
                    if x2 <= x1 or y2 <= y1:
                        continue
                    roi_img = frame[y1:y2, x1:x2]
                    if roi_img.ndim == 3:
                        gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                    else:
                        gray = roi_img
                    baseline[r["id"]] = float(np.mean(gray))
                    updated += 1
                print(f"Baseline set for {updated} ROIs.")
            else:
                # set baseline only for selected ROI
                r = roi_mgr.get(sid)
                if r:
                    x, y, w_, h_ = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])
                    x1 = max(0, min(x, frame.shape[1] - 1))
                    y1 = max(0, min(y, frame.shape[0] - 1))
                    x2 = max(0, min(x + w_, frame.shape[1]))
                    y2 = max(0, min(y + h_, frame.shape[0]))
                    if x2 > x1 and y2 > y1:
                        roi_img = frame[y1:y2, x1:x2]
                        if roi_img.ndim == 3:
                            gray = cv2.cvtColor(roi_img, cv2.COLOR_BGR2GRAY)
                        else:
                            gray = roi_img
                        baseline[sid] = float(np.mean(gray))
                        print(f"Baseline set for ROI#{sid} -> {baseline[sid]:.1f}")
                    else:
                        print("Selected ROI has invalid area.")
                else:
                    print("Selected ROI not found.")
        elif k == ord('r'):
            baseline.clear()
            print("All baselines cleared.")
        elif k == ord('v'):
            # save baseline to file
            save_baseline()
        elif k == ord('o'):
            # load baseline from file
            load_baseline()
        elif k == ord('p'):
            # print baseline table
            if baseline:
                for k_id, v in baseline.items():
                    print(f"ROI#{k_id} baseline: {v:.1f}")
            else:
                print("No baselines set.")

    roi_editor.detach_window()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
