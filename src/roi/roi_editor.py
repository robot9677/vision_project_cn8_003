import cv2
import time
import math
import numpy as np
from ui import ui_config as cfg
from ui import overlay_clean as overlay


class ROIEditor:
    """
    Enhanced ROI editor supporting:
      - Left-drag on empty area: create ROI (existing behavior)
      - Left-click inside ROI + drag: move ROI
      - Left-click near edges/corners + drag: resize ROI
      - Right-click inside ROI: delete ROI
      - Double-click inside ROI: rename ROI (terminal input)
    """
    HANDLE_RADIUS = 6
    EDGE_MARGIN = 8

    def __init__(self, roi_mgr, min_size=20):
        self.roi_mgr = roi_mgr
        self.min_size = min_size

        self._win_attached = False
        self._win_name = None

        # drag/create vars
        self.dragging = False
        self.creating = False
        self.x0 = self.y0 = self.x1 = self.y1 = 0

        # move/resize vars
        self.action = None  # None, 'move', 'resize'
        self.active_roi = None
        self.resize_dir = None  # ('l','r','t','b') combinations
        self.last_mouse = (0,0)
        self.rotate_start_angle = 0.0
        self.rotate_base_angle = 0.0

        # double-click detection
        self._last_click_time = 0
        self._double_click_interval = 0.35

        self.on_select_changed = lambda: None

    def _rotate_vec(self, vx, vy, angle_deg):
        th = math.radians(angle_deg)
        c = math.cos(th)
        s = math.sin(th)
        return (vx * c - vy * s, vx * s + vy * c)

    def _get_roi_top_geometry(self, r, handle_dist=30, label_dist=18):
        x, y, w, h = r["x"], r["y"], r["w"], r["h"]
        angle = float(r.get("angle", 0.0))

        cx = x + w / 2.0
        cy = y + h / 2.0

        # ROI 로컬 좌표계 기준 "상단"
        top_mid_local = (0.0, -h / 2.0)
        handle_local = (0.0, -(h / 2.0 + handle_dist))
        label_local = (0.0, -(h / 2.0 + label_dist))

        top_mid_dx, top_mid_dy = self._rotate_vec(top_mid_local[0], top_mid_local[1], angle)
        handle_dx, handle_dy = self._rotate_vec(handle_local[0], handle_local[1], angle)
        label_dx, label_dy = self._rotate_vec(label_local[0], label_local[1], angle)

        top_mid = (cx + top_mid_dx, cy + top_mid_dy)
        handle_pt = (cx + handle_dx, cy + handle_dy)
        label_pt = (cx + label_dx, cy + label_dy)

        return {
            "center": (int(cx), int(cy)),
            "top_mid": (int(top_mid[0]), int(top_mid[1])),
            "handle": (int(handle_pt[0]), int(handle_pt[1])),
            "label": (int(label_pt[0]), int(label_pt[1])),
        }
    
    def _get_rotate_handle_point(self, r):
        g = self._get_roi_top_geometry(r, handle_dist=30, label_dist=18)
        handle_x, handle_y = g["handle"]
        cx, cy = g["center"]
        return handle_x, handle_y, cx, cy

    def _hit_test(self, x, y):
        """Return roi dict and hit region: 'inside', 'edge', 'corner', or None"""
        rois = sorted(self.roi_mgr.list(), key=lambda r: (r["w"] * r["h"]))
        for r in rois:
            rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
            angle = float(r.get("angle", 0.0))

            cx = rx + rw / 2.0
            cy = ry + rh / 2.0
            box = cv2.boxPoints(((cx, cy), (rw, rh), angle)).astype(float)

            handle_x, handle_y, cxi, cyi = self._get_rotate_handle_point(r)

            # rotated ROI + rotate handle 포함 bounding box
            pts = np.vstack([box, np.array([[handle_x, handle_y]], dtype=float)])
            min_x = int(np.floor(np.min(pts[:, 0]))) - self.EDGE_MARGIN
            max_x = int(np.ceil(np.max(pts[:, 0]))) + self.EDGE_MARGIN
            min_y = int(np.floor(np.min(pts[:, 1]))) - self.EDGE_MARGIN
            max_y = int(np.ceil(np.max(pts[:, 1]))) + self.EDGE_MARGIN

            if not (min_x <= x <= max_x and min_y <= y <= max_y):
                continue

            # rotate handle 우선
            if abs(x - handle_x) <= 8 and abs(y - handle_y) <= 8:
                return r, "rotate", None

            # center
            if abs(x - cxi) <= 8 and abs(y - cyi) <= 8:
                return r, "center", None
                            
            inside = (rx <= x <= rx+rw and ry <= y <= ry+rh)
            # corners
            corners = {
                "tl": (rx, ry),
                "tr": (rx+rw, ry),
                "bl": (rx, ry+rh),
                "br": (rx+rw, ry+rh)
            }
            for name, (corner_x, corner_y) in corners.items():
                if abs(x-corner_x) <= self.EDGE_MARGIN and abs(y-corner_y) <= self.EDGE_MARGIN:
                    return r, "corner", name
            # edges
            if abs(x - rx) <= self.EDGE_MARGIN and ry <= y <= ry+rh:
                return r, "edge", "l"
            if abs(x - (rx+rw)) <= self.EDGE_MARGIN and ry <= y <= ry+rh:
                return r, "edge", "r"
            if abs(y - ry) <= self.EDGE_MARGIN and rx <= x <= rx+rw:
                return r, "edge", "t"
            if abs(y - (ry+rh)) <= self.EDGE_MARGIN and rx <= x <= rx+rw:
                return r, "edge", "b"
            if inside:
                return r, "inside", None
            return r, "near", None
        return None, None, None

    def _on_mouse(self, event, x, y, flags, param):
        t = time.time()
        if event == cv2.EVENT_LBUTTONDBLCLK:
            r, typ, sub = self._hit_test(x, y)
            if r:
                # select and rename via terminal input
                self.active_roi = r["id"]
                self.roi_mgr.select(r["id"])
                self.on_select_changed()
                try:
                    newname = input(f"Enter new name for ROI#{r['id']} (current='{r.get('name','')}'): ").strip()
                    if newname:
                        self.roi_mgr.update(r["id"], name=newname)
                        print("Renamed.")
                except Exception as e:
                    print("Rename cancelled or failed:", e)
            return

        if event == cv2.EVENT_LBUTTONDOWN:
            r, typ, sub = self._hit_test(x, y)
            self.last_mouse = (x,y)
            if r is None:
                # start creating a new ROI
                self.creating = True
                self.dragging = True
                self.action = None
                self.x0, self.y0 = x, y
                self.x1, self.y1 = x, y
            else:
                # clicked on existing ROI: decide move or resize
                self.active_roi = r["id"]
                self.roi_mgr.select(self.active_roi)
                self.on_select_changed()

                if typ in ("corner","edge"):
                    self.action = "resize"
                    self.resize_dir = sub
                elif typ == "rotate":
                    self.action = "rotate"
                    r_angle = float(r.get("angle", 0.0))
                    cx = r["x"] + r["w"] / 2.0
                    cy = r["y"] + r["h"] / 2.0
                    self.rotate_start_angle = math.degrees(math.atan2(y - cy, x - cx))
                    self.rotate_base_angle = r_angle
                elif typ == "center":
                    self.action = "move"
                else:
                    self.action = "move"

                self.dragging = True
            return

        if event == cv2.EVENT_MOUSEMOVE:
            if not self.dragging:
                return
            dx = x - self.last_mouse[0]
            dy = y - self.last_mouse[1]
            self.last_mouse = (x,y)
            if self.creating:
                self.x1, self.y1 = x, y
            elif self.action == "move" and self.active_roi is not None:
                r = self.roi_mgr.get(self.active_roi)
                if r:
                    nx = r["x"] + dx
                    ny = r["y"] + dy
                    self.roi_mgr.update(self.active_roi, x=nx, y=ny)
            elif self.action == "resize" and self.active_roi is not None:
                r = self.roi_mgr.get(self.active_roi)
                if r:
                    rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
                    nx, ny, nw, nh = rx, ry, rw, rh
                    dir = self.resize_dir
                    if dir == "l":
                        nx = rx + dx
                        nw = rw - dx
                    elif dir == "r":
                        nw = rw + dx
                    elif dir == "t":
                        ny = ry + dy
                        nh = rh - dy
                    elif dir == "b":
                        nh = rh + dy
                    # corners: handle names like 'tl','br' etc.
                    if dir in ("tl","tr","bl","br"):
                        if "l" in dir:
                            nx = rx + dx
                            nw = rw - dx
                        if "r" in dir:
                            nw = rw + dx
                        if "t" in dir:
                            ny = ry + dy
                            nh = rh - dy
                        if "b" in dir:
                            nh = rh + dy
                    # enforce min size
                    nw = max(self.min_size, int(nw))
                    nh = max(self.min_size, int(nh))
                    nx, ny, nw, nh = self.roi_mgr._clamp_rect(nx, ny, nw, nh)
                    self.roi_mgr.update(self.active_roi, x=nx, y=ny, w=nw, h=nh)
            elif self.action == "rotate" and self.active_roi is not None:
                r = self.roi_mgr.get(self.active_roi)
                if r:
                    cx = r["x"] + r["w"] / 2.0
                    cy = r["y"] + r["h"] / 2.0
                    cur_angle = math.degrees(math.atan2(y - cy, x - cx))
                    new_angle = self.rotate_base_angle + (cur_angle - self.rotate_start_angle)
                    self.roi_mgr.update(self.active_roi, angle=float(new_angle))
            return

        if event == cv2.EVENT_LBUTTONUP:
            if not self.dragging:
                return
            self.dragging = False
            if self.creating:
                x = min(self.x0, self.x1)
                y = min(self.y0, self.y1)
                w = abs(self.x1 - self.x0)
                h = abs(self.y1 - self.y0)
                if w >= self.min_size and h >= self.min_size:
                    self.roi_mgr.add(x, y, w, h)
                self.creating = False
            # finish move/resize
            self.action = None
            self.active_roi = None
            self.resize_dir = None
            self.rotate_start_angle = 0.0
            self.rotate_base_angle = 0.0
            return

        if event == cv2.EVENT_RBUTTONDOWN:
            # delete ROI if right-click inside
            r, typ, sub = self._hit_test(x, y)
            if r:
                rid = r["id"]
                self.roi_mgr.remove(rid)
                print(f"Deleted ROI#{rid}")
            return

    def attach_window(self, win_name):
        if self._win_attached and self._win_name == win_name:
            return
        self._win_name = win_name
        cv2.setMouseCallback(win_name, self._on_mouse)
        self._win_attached = True

    def detach_window(self):
        if self._win_attached and self._win_name is not None:
            cv2.setMouseCallback(self._win_name, lambda *args: None)
        self._win_attached = False
        self._win_name = None

    def update(self, vis_bgr):
        # draw handles and selection highlight with improved contrast
        sel_id = self.roi_mgr.selected_id

        Himg, Wimg = vis_bgr.shape[:2]

        overlay.draw_origin_axes(vis_bgr, origin=(40, 60), axis_len=80)
        sel = self.roi_mgr.get_selected()

        parent_roi = None
        if sel is not None:
            sx, sy, sw, sh = sel["x"], sel["y"], sel["w"], sel["h"]
            parents = []
            for r in self.roi_mgr.list():
                if r["id"] == sel["id"]:
                    continue
                rx, ry, rw, rh = r["x"], r["y"], r["w"], r["h"]
                if sx >= rx and sy >= ry and sx + sw <= rx + rw and sy + sh <= ry + rh:
                    parents.append(r)
            if parents:
                parent_roi = min(parents, key=lambda r: r["w"] * r["h"])

        overlay.draw_selected_roi_info(vis_bgr, sel, parent_roi)

        # 1) optional: darken whole image a bit to make ROIs pop
        alpha = 0.1   # 0 = no darken, 0.35 = mild darken (조절 가능)
        if alpha > 0:
            ovl = vis_bgr.copy()
            dark = ovl.copy()
            overlay.draw_rect(dark, (0,0), (Wimg, Himg), color=(0,0,0), fill=True)
            cv2.addWeighted(dark, alpha, vis_bgr, 1.0 - alpha, 0, vis_bgr)

        # 2) draw each ROI with a black outline then bright colored inner stroke for visibility
        default_color = cfg.COLOR_ROI   # from ui config
        selected_color = cfg.COLOR_ROI_ACTIVE
        outline_color = (0,0,0)
        rect_thick_outline = 4
        rect_thick_inner = 2
        font = cfg.FONT
        font_scale = cfg.FONT_SCALE
        text_offset = 8

        # --- Replace the for-loop drawing block with this ---
        # bright palette (ensure high intensity values)
        palette = [
            (0,255,255),   # yellow
            (0,128,255),   # orange
            (255,0,0),     # blue-ish (BGR)
            (0,255,0),     # green
            (255,0,255),   # magenta
            (255,255,0),   # cyan
            (180,255,180), # pale green
        ]

        font = cfg.FONT
        font_scale = cfg.FONT_SCALE

        # Reduce darkness of overlay if present earlier in function:
        # if you have `alpha = 0.35` above, reduce to 0.12 (or 0 to disable)
        # alpha = 0.12

        for idx, r in enumerate(self.roi_mgr.list()):
            x, y, w, h = int(r["x"]), int(r["y"]), int(r["w"]), int(r["h"])

            # pick bright color
            color = palette[idx % len(palette)]
            # non-selected: colored thick rect
            angle = float(r.get("angle", 0.0))
            cx = x + w / 2.0
            cy = y + h / 2.0
            box = cv2.boxPoints(((cx, cy), (w, h), angle))
            box = box.astype(int)

            if r["id"] != sel_id:
                cv2.polylines(vis_bgr, [box], True, color, 1, lineType=cv2.LINE_AA)
            else:
                # selected: rotated outer/inner box
                box_outer = cv2.boxPoints(((cx, cy), (w, h), angle)).astype(int)
                box_inner = cv2.boxPoints(((cx, cy), (max(1, w - 2), max(1, h - 2)), angle)).astype(int)

                cv2.polylines(vis_bgr, [box_outer], True, (255, 255, 255), 2, lineType=cv2.LINE_AA)
                cv2.polylines(vis_bgr, [box_inner], True, color, 1, lineType=cv2.LINE_AA)

                # center cross
                cxi = int(cx)
                cyi = int(cy)
                cv2.line(vis_bgr, (cxi - 6, cyi), (cxi + 6, cyi), (0, 255, 255), 1, lineType=cv2.LINE_AA)
                cv2.line(vis_bgr, (cxi, cyi - 6), (cxi, cyi + 6), (0, 255, 255), 1, lineType=cv2.LINE_AA)

                # rotated corner handles
                for hx, hy in box_outer:
                    cv2.circle(vis_bgr, (int(hx), int(hy)), self.HANDLE_RADIUS, (255, 255, 255), -1, lineType=cv2.LINE_AA)

                # rotation handle: top edge midpoint -> outward
                g = self._get_roi_top_geometry(r, handle_dist=30, label_dist=18)
                top_mid_x, top_mid_y = g["top_mid"]
                handle_x, handle_y = g["handle"]

                cv2.line(
                    vis_bgr,
                    (top_mid_x, top_mid_y),
                    (handle_x, handle_y),
                    (0, 255, 255),
                    1,
                    lineType=cv2.LINE_AA
                )
                cv2.circle(vis_bgr, (handle_x, handle_y), 4, (0, 255, 255), -1, lineType=cv2.LINE_AA)

            # label with shadow for contrast
            label = f'{r.get("name","ROI")}'

            g = self._get_roi_top_geometry(r, handle_dist=30, label_dist=18)
            label_x, label_y = g["label"]

            tx = int(label_x - 18)
            ty = int(label_y - 10)

            # shadow
            overlay.draw_text(vis_bgr, label,(tx + 1, ty + 1),color=(0, 0, 0),scale=cfg.FONT_SCALE - 0.1,thickness=3,align='lt')

            # main text
            overlay.draw_text(vis_bgr, label,(tx, ty),color=cfg.COLOR_TEXT,scale=cfg.FONT_SCALE - 0.1,thickness=cfg.FONT_THICK,align='lt')
        # --- end replacement ---

        # 3) during creation, draw preview (keep bright color)
        if self.creating:
            x = min(self.x0, self.x1)
            y = min(self.y0, self.y1)
            w = abs(self.x1 - self.x0)
            h = abs(self.y1 - self.y0)
            overlay.draw_rect(vis_bgr, (x, y), (x + w, y + h), color=(0,255,255), thickness=2)
