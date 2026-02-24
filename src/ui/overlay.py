# src/ui/overlay.py
import cv2
from typing import Dict, Any, Optional, List, Tuple

# Central UI config used by main_vp; these may be overridden by UI_CONFIG in main if desired.
UI = {
    "font": cv2.FONT_HERSHEY_SIMPLEX,
    "font_scale": 0.55,
    "thickness": 1,
    "label_bg_alpha": 0.45,
    "label_text_color": (255, 220, 20),
    "roi_ok_color": (0, 200, 0),
    "roi_ng_color": (0, 0, 200),
    "roi_default_color": (0, 200, 200),
    "roi_highlight_color": (0, 255, 255),
    "status_bar_height": 40,
    "control_bar_height": 72,
    "overlay_margin": 6,
    "line_spacing": 4,
}

def put_text_with_bg(img, text: str, pos: Tuple[int,int], font_scale=None, thickness=None, color=None):
    font = UI["font"]
    if font_scale is None:
        font_scale = UI["font_scale"]
    if thickness is None:
        thickness = UI["thickness"]
    if color is None:
        color = UI["label_text_color"]
    (tw, th), baseline = cv2.getTextSize(text, font, font_scale, thickness)
    x, y = pos
    # background rect
    bx1 = max(2, x-4); by1 = max(2, y-th-4)
    bx2 = x + tw + 4; by2 = y + 4
    overlay = img.copy()
    cv2.rectangle(overlay, (bx1, by1), (bx2, by2), (0,0,0), -1)
    cv2.addWeighted(overlay, UI["label_bg_alpha"], img, 1 - UI["label_bg_alpha"], 0, img)
    cv2.putText(img, text, (x, y), font, font_scale, color, thickness, cv2.LINE_AA)

def draw_status_bar(img, text: str):
    h, w = img.shape[:2]
    bar_h = UI["status_bar_height"]
    overlay = img.copy()
    cv2.rectangle(overlay, (0, 0), (w, bar_h), (0, 0, 0), -1)
    cv2.addWeighted(overlay, UI["label_bg_alpha"], img, 1 - UI["label_bg_alpha"], 0, img)
    put_text_with_bg(img, text, (10, int(bar_h*0.72)), font_scale=0.75, thickness=2)

def draw_control_bar(img, buttons: List[Dict[str,Any]]) -> List[Dict[str, Any]]:
    h, w = img.shape[:2]
    bar_h = UI["control_bar_height"]
    overlay = img.copy()
    y0 = h - bar_h
    cv2.rectangle(overlay, (0, y0), (w, h), (30,30,30), -1)
    cv2.addWeighted(overlay, 0.85, img, 0.15, 0, img)
    n = len(buttons)
    if n == 0:
        return buttons
    min_btn_w = 100
    btn_w = max(min_btn_w, int(w / n))
    btn_h = bar_h - 16
    x = 8
    for b in buttons:
        if b.get("rect") is None:
            x1 = x; y1 = y0 + 8
            x2 = min(w - 8, x1 + btn_w - 8); y2 = y1 + btn_h
            x = x2 + 8
            b["rect"] = (int(x1), int(y1), int(x2), int(y2))
        x1,y1,x2,y2 = b["rect"]
        bg = b.get("color", (60,60,60))
        cv2.rectangle(img, (x1,y1), (x2,y2), bg, -1)
        cv2.rectangle(img, (x1,y1), (x2,y2), (180,180,180), 1)
        label = b.get("label", b.get("id",""))
        (tw, th), _ = cv2.getTextSize(label, UI["font"], 0.7, 2)
        tx = x1 + max(8, (x2-x1 - tw)//2)
        ty = y1 + max(24, (y2-y1 + th)//2)
        cv2.putText(img, label, (tx, ty), UI["font"], 0.7, (255,255,255), 2, cv2.LINE_AA)
    return buttons

def draw_rois_clean(img, roi_list, highlight_id=None, roi_results: Optional[Dict[str,Any]] = None):
    """
    roi_list: list of dicts {'id','name','x','y','w','h'} or object with .rois attribute
    roi_results: dict keyed by str(id) or int(id)
    """
    # adapt roi_list if object
    rois = None
    if isinstance(roi_list, dict):
        rois = roi_list.get("rois", [])
    elif hasattr(roi_list, "rois"):
        rois = getattr(roi_list, "rois")
    elif callable(getattr(roi_list, "list", None)):
        rois = roi_list.list()
    else:
        rois = list(roi_list)

    # draw rectangles + compute labels
    labels = []
    for r in rois:
        try:
            rid = r.get("id"); x = int(r["x"]); y = int(r["y"]); w = int(r["w"]); h = int(r["h"])
        except Exception:
            continue
        # color default
        color = UI["roi_default_color"]
        lines = [f"ROI{rid}"]
        if roi_results:
            entry = None
            if isinstance(roi_results, dict):
                entry = roi_results.get(str(rid)) or roi_results.get(int(rid))
            if entry:
                if hasattr(entry, "metrics"):
                    metrics = entry.metrics
                    ok = getattr(entry, "ok", None)
                    reason = getattr(entry, "reason", "")
                elif isinstance(entry, dict):
                    metrics = entry.get("metrics", {})
                    ok = entry.get("ok", None)
                    reason = entry.get("reason", "")
                else:
                    metrics = {}; ok = None; reason = ""
                if ok is True:
                    color = UI["roi_ok_color"]
                    lines[0] = f"ROI{rid} OK"
                elif ok is False:
                    color = UI["roi_ng_color"]
                    lines[0] = f"ROI{rid} NG:{reason}"
                meanv = metrics.get("mean", metrics.get("mean_raw", None))
                scorev = metrics.get("score", None)
                p = []
                if meanv is not None:
                    p.append(f"m:{meanv:.1f}")
                if scorev is not None:
                    p.append(f"s:{scorev:.2f}")
                if p:
                    lines.append(" ".join(p))
        # highlight
        thickness = 2
        if highlight_id is not None and rid == highlight_id:
            thickness = 3
            color = UI["roi_highlight_color"]
        cv2.rectangle(img, (x,y), (x+w, y+h), color, thickness)
        anchor = 'top' if (y - UI["overlay_margin"] - 40) > 0 else 'bottom'
        labels.append({"rect":(x,y,w,h), "lines": lines, "anchor": anchor})

    # draw labels top then bottom, sorted
    def sort_key(l): return (l["rect"][1], l["rect"][0])
    top = [L for L in labels if L["anchor"]=="top"]; top.sort(key=sort_key)
    bottom = [L for L in labels if L["anchor"]=="bottom"]; bottom.sort(key=sort_key)
    for L in top + bottom:
        x,y,w,h = L["rect"]; lines = L["lines"]
        fs = UI["font_scale"]; th = UI["thickness"]
        sizes = [cv2.getTextSize(s, UI["font"], fs, th)[0] for s in lines]
        max_w = max(sz[0] for sz in sizes) if sizes else 0
        total_h = sum(sz[1] for sz in sizes) + (len(lines)-1)*UI["line_spacing"]
        if L["anchor"] == "top":
            tx = x; ty_base = y - UI["overlay_margin"]
            start_y = ty_base - total_h + sizes[0][1]
        else:
            tx = x; start_y = y + h + UI["overlay_margin"] + sizes[0][1]
        bx1 = tx - 4; by1 = int(start_y - sizes[0][1] - 4)
        bx2 = tx + max_w + 8; by2 = int(start_y + total_h + 4)
        ih, iw = img.shape[:2]
        bx1 = max(2, bx1); by1 = max(2, by1); bx2 = min(iw-2, bx2); by2 = min(ih-2, by2)
        overlay = img.copy()
        cv2.rectangle(overlay, (bx1,by1), (bx2,by2), (0,0,0), -1)
        cv2.addWeighted(overlay, UI["label_bg_alpha"], img, 1 - UI["label_bg_alpha"], 0, img)
        cur_y = int(start_y)
        for s in lines:
            cv2.putText(img, s, (tx, cur_y), UI["font"], fs, UI["label_text_color"], th, cv2.LINE_AA)
            cur_y += cv2.getTextSize(s, UI["font"], fs, th)[0][1] + UI["line_spacing"]