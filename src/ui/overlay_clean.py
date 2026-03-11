# src/ui/overlay.py
import cv2
import numpy as np
from ui import ui_config as cfg
from PIL import ImageFont, ImageDraw, Image

# --- basic drawing helpers ---
def draw_text(img, text, pos, color=None, scale=None, thickness=None, align="lt"):
    if color is None:
        color = cfg.COLOR_TEXT
    if scale is None:
        scale = cfg.FONT_SCALE
    if thickness is None:
        thickness = cfg.FONT_THICK

    (tw, th), baseline = cv2.getTextSize(str(text), cfg.FONT, scale, thickness)
    x, y = int(pos[0]), int(pos[1])

    if align == "ct":
        x = int(x - tw / 2)
    elif align == "rt":
        x = int(x - tw)

    # align vertical: use baseline approx (putText uses bottom-left)
    y_draw = int(y + th / 2)
    cv2.putText(img, str(text), (x, y_draw), cfg.FONT, scale, color, int(thickness), cfg.LINE_TYPE)

def draw_text_kr(img, text, pos, size=28, color=(255,255,255)):
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"

    img_pil = Image.fromarray(img)
    draw = ImageDraw.Draw(img_pil)
    font = ImageFont.truetype(font_path, size)

    draw.text(pos, text, font=font, fill=color)

    return np.array(img_pil)

def draw_rect(img, pt1, pt2, color=None, thickness=1, fill=False):
    if color is None:
        color = cfg.COLOR_TEXT
    x1, y1 = int(pt1[0]), int(pt1[1])
    x2, y2 = int(pt2[0]), int(pt2[1])
    if fill:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, -1)
    else:
        cv2.rectangle(img, (x1, y1), (x2, y2), color, int(thickness), lineType=cfg.LINE_TYPE)


# --- higher-level UI elements ---
def draw_status_bar(img, text):
    h, w = img.shape[:2]
    bar_h = cfg.STATUS_HEIGHT if hasattr(cfg, "STATUS_HEIGHT") else 40
    # dark background
    dark = img.copy()
    draw_rect(dark, (0, 0), (w, bar_h), color=cfg.STATUS_BG if hasattr(cfg, "STATUS_BG") else (0,0,0), fill=True)
    alpha = 0.6
    cv2.addWeighted(dark, alpha, img, 1.0 - alpha, 0, img)
    draw_text(img, text, (cfg.MARGIN if hasattr(cfg, "MARGIN") else 8, int(bar_h/2)+6), color=cfg.COLOR_TEXT, scale=cfg.FONT_SCALE, thickness=cfg.FONT_THICK, align="lt")


def draw_rois(img, rois=None, active_id=None, roi_results=None, show_only_selected=False, compact=False, **kwargs):
    """
    rois: list of dicts {'id':int, 'label':str, 'rect':(x,y,w,h)} OR objects with x,y,w,h,name,id
    active_id: roi id to highlight
    roi_results: optional dict keyed by roi id or str(id)
    show_only_selected: bool
    """
    if rois is None:
        # if older code passed a roi_mgr, try to adapt
        rois = []

    h_img, w_img = img.shape[:2]
    base_font = cfg.FONT
    base_font_scale = cfg.FONT_SCALE
    base_thickness = cfg.FONT_THICK
    line_spacing = 4

    roi_text_color = (0, 255, 0)
    roi_text_scale = 0.4
    roi_text_thickness = 1

    for idx, r in enumerate(rois):
        # normalize roi dict/object to {id, x,y,w,h, name/label}
        if isinstance(r, dict):
            roi_id = r.get("id")
            label = r.get("label") or r.get("name") or f"ROI{roi_id}"
            x = int(r.get("rect", r.get("bbox", (r.get("x",0), r.get("y",0), r.get("w",0), r.get("h",0))))[0])
            y = int(r.get("rect", r.get("bbox", (r.get("x",0), r.get("y",0), r.get("w",0), r.get("h",0))))[1])
            w = int(r.get("rect", r.get("bbox", (r.get("x",0), r.get("y",0), r.get("w",0), r.get("h",0))))[2])
            h = int(r.get("rect", r.get("bbox", (r.get("x",0), r.get("y",0), r.get("w",0), r.get("h",0))))[3])
        else:
            # object-like
            roi_id = getattr(r, "id", None)
            label = getattr(r, "name", getattr(r, "label", f"ROI{roi_id}"))
            x = int(getattr(r, "x", 0)); y = int(getattr(r, "y", 0))
            w = int(getattr(r, "w", 0)); h = int(getattr(r, "h", 0))

        if show_only_selected and active_id is not None and roi_id != active_id:
            continue

        # determine color by default
        color = cfg.COLOR_ROI if hasattr(cfg, "COLOR_ROI") else (0,200,200)
        thickness = base_thickness

        # check roi_results for status/metrics
        rid_str = str(roi_id)
        if roi_results is not None:
            rv = None
            if isinstance(roi_results, dict):
                rv = roi_results.get(rid_str) if rid_str in roi_results else roi_results.get(roi_id)
            if rv is not None:
                ok = rv.get("ok") if isinstance(rv, dict) else getattr(rv, "ok", None)
                reason = rv.get("reason") if isinstance(rv, dict) else getattr(rv, "reason", "")
                metrics = rv.get("metrics") if isinstance(rv, dict) else getattr(rv, "metrics", None)
                if ok is True:
                    color = cfg.COLOR_OK
                elif ok is False:
                    color = cfg.COLOR_NG
                else:
                    # keep default
                    pass

        if roi_id == active_id:
            thickness = thickness + 1
            # override active color if configured
            color = cfg.COLOR_ROI_ACTIVE if hasattr(cfg, "COLOR_ROI_ACTIVE") else color

        # draw rectangle
        # ----- rotated rectangle -----
        angle = float(r.get("angle", 0.0)) if isinstance(r, dict) else float(getattr(r, "angle", 0.0))

        cx = x + w / 2
        cy = y + h / 2

        rect = ((cx, cy), (w, h), angle)

        box = cv2.boxPoints(rect)
        box = np.int32(box)

        cv2.polylines(img, [box], True, color, thickness, lineType=cv2.LINE_AA)

        # center point
        cv2.circle(img, (int(cx), int(cy)), 3, color, -1, lineType=cv2.LINE_AA)

        # prepare label lines
        line1 = f"{label}#{roi_id}" if roi_id is not None else label

        angle = float(r.get("angle", 0.0)) if isinstance(r, dict) else float(getattr(r, "angle", 0.0))

        if compact:
            line2 = ""
        else:
            line2 = f"x:{x} y:{y} w:{w} h:{h} a:{angle:.1f}"

        line3 = "SELECTED" if (roi_id == active_id and not compact) else ""

        # metric summary (optional)
        if roi_results is not None:
            rv = roi_results.get(rid_str) if isinstance(roi_results, dict) else None
            if rv is None and isinstance(roi_results, dict):
                rv = roi_results.get(roi_id)
            if rv:
                metrics = rv.get("metrics") if isinstance(rv, dict) else getattr(rv, "metrics", None)

                def _fmt(k, v):
                    try:
                        fv = float(v)
                        # key별 기본 포맷
                        if k in ("score",):
                            return f"s:{fv:.2f}"
                        if k in ("white_ratio",):
                            return f"wr:{fv:.2f}"
                        if k in ("edge_energy", "lap_var", "laplacian_var"):
                            return f"e:{fv:.1f}"
                        if k in ("mean", "mean_gray", "mean_raw"):
                            return f"m:{fv:.1f}"
                        return f"{k}:{fv:.2f}"
                    except Exception:
                        # 문자열류(예: qr_data)
                        sv = str(v)
                        if k in ("qr_data", "barcode", "text"):
                            sv = sv[:12]  # 너무 길면 잘라서
                            return f"qr:{sv}"
                        return f"{k}:{sv[:12]}"

                parts = []
                if isinstance(metrics, dict):
                    # 표시 우선순위
                    order = ["mean", "score", "white_ratio", "edge_energy", "qr_data"]
                    for k in order:
                        if k in metrics and metrics[k] is not None:
                            parts.append(_fmt(k, metrics[k]))
                        if len(parts) >= 2:   # line2는 2개까지만
                            break

                if parts:
                    line2 = " ".join(parts)

        lines = [line1, line2] + ([line3] if line3 else [])

        # text sizing
        sizes = [cv2.getTextSize(s, base_font, base_font_scale, base_thickness)[0] for s in lines]
        heights = [cv2.getTextSize(s, base_font, base_font_scale, base_thickness)[0][1] for s in lines]
        max_w = max(sz[0] for sz in sizes) if sizes else 0
        total_h = sum(heights) + max(0, (len(lines)-1)) * line_spacing

        margin = 10

        # rotated ROI 기준 상단 방향 라벨 anchor
        top_mid_local = np.array([0.0, -h / 2.0], dtype=float)
        label_local = np.array([0.0, -(h / 2.0 + margin + total_h)], dtype=float)

        th = np.radians(angle)
        c = np.cos(th)
        s = np.sin(th)

        rot = np.array([[c, -s], [s, c]], dtype=float)

        label_pt = np.array([cx, cy], dtype=float) + rot @ label_local

        bg_x1 = int(label_pt[0] - max_w / 2 - 4)
        bg_y1 = int(label_pt[1] - 4)
        bg_x2 = bg_x1 + max_w + 8
        bg_y2 = bg_y1 + total_h + 8

        # clamp
        if bg_x1 < 2:
            bg_x2 += (2 - bg_x1)
            bg_x1 = 2
        if bg_x2 > w_img - 2:
            shift = bg_x2 - (w_img - 2)
            bg_x1 -= shift
            bg_x2 -= shift
        if bg_y1 < 2:
            bg_y2 += (2 - bg_y1)
            bg_y1 = 2
        if bg_y2 > h_img - 2:
            shift = bg_y2 - (h_img - 2)
            bg_y1 -= shift
            bg_y2 -= shift

        top_text_y = bg_y1 + 4 + heights[0]

        # draw semi-transparent background
        if compact:
            lines2 = [s for s in lines if s]

            # EDIT와 동일하게 수평 텍스트 + ROI 상단 기준 위치
            tx = int(x-20)
            ty = int(y - 20 if y > 20 else y + h + 18)

            for i, t in enumerate(lines2):
                draw_text(
                    img,
                    t,
                    (tx, ty + (i * 14)),
                    color=roi_text_color,
                    scale=roi_text_scale,
                    thickness=roi_text_thickness,
                    align="lt",
                )
        else:
            temp = img.copy()
            draw_rect(temp, (bg_x1, bg_y1), (bg_x2, bg_y2), color=(0,0,0), fill=True)
            draw_rect(img, (bg_x1, bg_y1), (bg_x2, bg_y2), color=(50,50,50), thickness=1)
            alpha = 0.4
            cv2.addWeighted(temp, alpha, img, 1-alpha, 0, img)

            # draw text lines
            cur_y = top_text_y + heights[0]
            for i, t in enumerate(lines):
                tx = bg_x1 + 4
                draw_text(img, t, (tx, cur_y), color=roi_text_color, scale=roi_text_scale, thickness=roi_text_thickness, align="lt")
                if i+1 < len(lines):
                    cur_y += heights[i+1] + line_spacing


def draw_overall_banner(img, overall_ok, info=None):
    h, w = img.shape[:2]

    # ----- 위치 정책 (여기만 수정하면 전체 위치 변경됨) -----
    POS = {
        "overall": ("ct", (0, 25)),      # center-top
        "debug":   ("rb", (12, 88)),     # right-bottom (버튼바 피해서)
    }
    # -------------------------------------------------------

    ng = int(info.get("ng", 0)) if isinstance(info, dict) else 0
    total = int(info.get("total", 0)) if isinstance(info, dict) else 0

    text  = f"OVERALL: {'OK' if overall_ok else 'NG'} ({ng}/{total} NG)"
    color = cfg.COLOR_OK if overall_ok else cfg.COLOR_NG

    # ---- OVERALL (중앙 상단) ----
    align, (mx, my) = POS["overall"]
    x = w // 2 + mx
    y = my
    draw_text(img, text, (x, y-10), color=color, scale=0.8, thickness=2, align=align)

    # ---- debug (우측 하단) ----
    if isinstance(info, dict):
        parts = []

        if "norm_gain" in info:
            try:
                parts.append(f"gain={float(info['norm_gain']):.2f}")
            except Exception:
                parts.append(f"gain={info['norm_gain']}")

        if "dx" in info:
            parts.append(f"dx={int(info['dx'])}")

        if "dy" in info:
            parts.append(f"dy={int(info['dy'])}")

        if parts:
            dbg = "  ".join(parts)

            align, (mx, my) = POS["debug"]
            x = w - mx
            y = h - my

            draw_text(img, dbg, (x-250, y+2), color=cfg.COLOR_TEXT, scale=0.6, thickness=1, align=align)


def draw_origin_axes(img, origin=(40, 60), axis_len=80):
    ox, oy = int(origin[0]), int(origin[1])

    # origin point
    cv2.circle(img, (ox, oy), 3, (0, 255, 255), -1, lineType=cv2.LINE_AA)

    # X axis
    cv2.arrowedLine(img, (ox, oy), (ox + axis_len, oy), (0, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.12)
    draw_text(img, "X+", (ox + axis_len + 8, oy), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

    # Y axis
    cv2.arrowedLine(img, (ox, oy), (ox, oy + axis_len), (0, 255, 255), 1, line_type=cv2.LINE_AA, tipLength=0.12)
    draw_text(img, "Y+", (ox - 4, oy + axis_len + 14), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

    # origin label
    draw_text(img, f"(0,0)", (ox + 6, oy - 10), color=(0, 255, 255), scale=0.5, thickness=1, align="lt")

def draw_selected_roi_info(img, roi, parent_roi=None):
    if not roi:
        return

    x = int(roi.get("x", 0))
    y = int(roi.get("y", 0))
    w = int(roi.get("w", 0))
    h = int(roi.get("h", 0))
    rid = roi.get("id", "")

    angle = float(roi.get("angle", 0.0))
    cx = int(x + w / 2)
    cy = int(y + h / 2)

    text = f"Selected ROI: {rid}  x:{x} y:{y} w:{w} h:{h} a:{angle:.1f}  cx:{cx} cy:{cy}"

    if parent_roi and parent_roi.get("id") != rid:
        px = int(parent_roi.get("x", 0))
        py = int(parent_roi.get("y", 0))
        dx = x - px
        dy = y - py
        text += f"  rel:{dx},{dy}"
        
    draw_text(img, text, (140, 20), color=(0, 255, 255), scale=0.6, thickness=1, align="lt")

def draw_control_bar(img, buttons):
    h, w = img.shape[:2]
    bar_h = 72
    temp = img.copy()
    y0 = h - bar_h
    draw_rect(temp, (0, y0), (w, h), color=(30, 30, 30), fill=True)
    cv2.addWeighted(temp, 0.85, img, 0.15, 0, img)

    n = len(buttons)
    if n == 0:
        return buttons
    min_btn_w = 100
    btn_w = max(min_btn_w, int(w / n))
    btn_h = bar_h - 16
    x = 8
    new_buttons = []
    for i, b in enumerate(buttons):
        if b.get("rect") is None:
            x1 = x
            y1 = y0 + 8
            x2 = min(w - 8, x1 + btn_w - 8)
            y2 = y1 + btn_h
            x = x2 + 8
            b["rect"] = (int(x1), int(y1), int(x2), int(y2))
        else:
            x1,y1,x2,y2 = b["rect"]
            b["rect"] = (int(x1), int(y1), int(x2), int(y2))

        x1,y1,x2,y2 = b["rect"]
        bg_color = b.get("color", (60,60,60))
        draw_rect(img, (x1, y1), (x2, y2), color=bg_color, fill=True)
        draw_rect(img, (x1, y1), (x2, y2), color=(180,180,180), thickness=1)
        label = b.get("label", b.get("id",""))
        (tw, th), baseline = cv2.getTextSize(label, cfg.FONT, 0.7, 2)
        tx = x1 + max(8, (x2-x1 - tw)//2)
        ty = y1 + max(24, (y2-y1 + th)//2)
        draw_text(img, label, (tx, ty), color=cfg.COLOR_TEXT, scale=0.7, thickness=2)
        new_buttons.append(b)
    return new_buttons