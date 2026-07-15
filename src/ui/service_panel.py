import hashlib
import os
import time
from typing import Any, Dict, List, Optional, Tuple

import cv2
import numpy as np


CMD_NAMES = {
    0: "NONE",
    1: "PREPARE",
    2: "INSPECT",
    3: "RESET",
    8: "SHUTDOWN",
    9: "EMERGENCY",
}
STATUS_NAMES = {
    0: "READY",
    1: "BUSY",
    2: "DONE",
    3: "ERROR",
    8: "SHUTDOWN",
}
RESULT_NAMES = {
    0: "NONE",
    1: "OK",
    2: "NG",
}
ERROR_NAMES = {
    0: "NO ERROR",
    10: "CAMERA INIT",
    11: "CAMERA FRAME",
    20: "LIGHT START",
    21: "LIGHT COMM",
    30: "RECIPE LOAD",
    31: "ROI CONFIG",
    40: "INSPECTION",
    41: "INSPECT EXCEPTION",
    50: "COMMAND SEQUENCE",
    51: "EMERGENCY STOP",
    60: "VISION RESET",
    70: "PLC SERIAL OPEN",
    71: "PLC COMM",
    90: "SHUTDOWN",
}


def _safe_int(value: Any, default: int = 0) -> int:
    try:
        return int(value)
    except Exception:
        return int(default)


def _normalize_cv_key(key: Any) -> int:
    """Normalize cv2.waitKeyEx() values across GTK/Qt/NoMachine.

    Some Linux/remote-desktop combinations return printable keys as
    0x100000 + ASCII.  Comparing that raw value directly makes every
    password key, Enter and Escape appear unresponsive.
    """
    try:
        raw = int(key)
    except Exception:
        return -1

    if raw < 0:
        return raw

    # X11/GTK keysyms commonly returned by waitKeyEx().
    aliases = {
        65307: 27,   # Escape
        65293: 13,   # Return
        65421: 13,   # Keypad Enter
        65288: 8,    # BackSpace
        65535: 127,  # Delete
    }

    if raw in aliases:
        return aliases[raw]

    low16 = raw & 0xFFFF
    if low16 in aliases:
        return aliases[low16]

    # OpenCV waitKeyEx() may preserve an implementation-specific high word.
    # Only strip it when a high word is actually present so X11 arrow keys
    # such as 65361 are not misread as printable ASCII.
    if raw >= 0x10000:
        low8 = raw & 0xFF
        if low8 in (8, 10, 13, 27, 127) or 32 <= low8 <= 126:
            return low8

    return raw


def _age_text(epoch: Any) -> str:
    try:
        age = max(0.0, time.time() - float(epoch))
    except Exception:
        return "-"
    if age < 10.0:
        return f"{age:.1f}s"
    if age < 60.0:
        return f"{age:.0f}s"
    return f"{age / 60.0:.1f}m"


def _ellipsize(text: Any, max_chars: int) -> str:
    text = str(text or "")
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 3)] + "..."


def _put_text(
    img: np.ndarray,
    text: str,
    x: int,
    y: int,
    scale: float = 0.46,
    color: Tuple[int, int, int] = (230, 230, 230),
    thickness: int = 1,
):
    cv2.putText(
        img,
        str(text),
        (int(x), int(y)),
        cv2.FONT_HERSHEY_SIMPLEX,
        float(scale),
        color,
        int(thickness),
        cv2.LINE_AA,
    )


def _draw_button(
    img: np.ndarray,
    rect: Tuple[int, int, int, int],
    label: str,
    enabled: bool,
    accent: Tuple[int, int, int],
):
    x1, y1, x2, y2 = rect
    fill = accent if enabled else (58, 58, 58)
    cv2.rectangle(img, (x1, y1), (x2, y2), fill, -1)
    cv2.rectangle(img, (x1, y1), (x2, y2), (170, 170, 170), 1)
    color = (255, 255, 255) if enabled else (120, 120, 120)
    (tw, th), _ = cv2.getTextSize(
        label, cv2.FONT_HERSHEY_SIMPLEX, 0.43, 1
    )
    tx = x1 + max(4, (x2 - x1 - tw) // 2)
    ty = y1 + max(th + 4, (y2 - y1 + th) // 2)
    _put_text(img, label, tx, ty, 0.43, color, 1)


class ServicePanel:
    """Password-protected in-window PLC/vision service panel."""

    def __init__(self, cfg: Optional[Dict[str, Any]] = None):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.panel_width = max(420, min(760, int(cfg.get("panel_width", 560))))
        self.opacity = min(1.0, max(0.70, float(cfg.get("opacity", 0.96))))
        self.password_sha256 = str(cfg.get("password_sha256", "")).strip().lower()
        self.default_password_notice = bool(cfg.get("default_password_notice", True))
        self.lock_on_close = bool(cfg.get("lock_on_close", True))
        self.show_roi_debug = bool(cfg.get("show_roi_debug", True))
        self.rx_tx_rows = max(2, min(8, int(cfg.get("rx_tx_rows", 4))))
        self.refresh_hz = max(1.0, min(15.0, float(cfg.get("refresh_hz", 5.0))))
        self.roi_debug_hz = max(0.2, min(5.0, float(cfg.get("roi_debug_hz", 1.0))))

        self.visible = False
        self.auth_pending = False
        self.password_buffer = ""
        self.auth_error = ""
        self.auth_error_until = 0.0
        self._buttons: List[Dict[str, Any]] = []
        self._auth_buttons: List[Dict[str, Any]] = []
        self._action: Optional[str] = None
        self._message = ""
        self._message_until = 0.0
        self._panel_rect: Optional[Tuple[int, int, int, int]] = None
        self._roi_resize_cache_key = None
        self._roi_resize_cache = None

    def is_modal(self) -> bool:
        return bool(self.auth_pending)

    def request_toggle(self):
        if not self.enabled:
            self.set_message("Service panel is disabled", 2.0)
            return
        if self.visible:
            self.visible = False
            self.auth_pending = False
            self.password_buffer = ""
            self._auth_buttons = []
            if self.lock_on_close:
                self._buttons = []
            return
        self.auth_pending = True
        self.password_buffer = ""
        self.auth_error = ""
        self._auth_buttons = []

    def close(self):
        self.visible = False
        self.auth_pending = False
        self.password_buffer = ""
        self._buttons = []
        self._auth_buttons = []

    def set_message(self, text: str, seconds: float = 2.5):
        self._message = str(text or "")
        self._message_until = time.time() + max(0.1, float(seconds))

    def pop_action(self) -> Optional[str]:
        action = self._action
        self._action = None
        return action

    def _password_matches(self, password: str) -> bool:
        if not self.password_sha256:
            return False
        digest = hashlib.sha256(password.encode("utf-8")).hexdigest().lower()
        return digest == self.password_sha256

    def _cancel_auth(self):
        self.auth_pending = False
        self.password_buffer = ""
        self.auth_error = ""
        self._auth_buttons = []

    def _submit_password(self):
        if self._password_matches(self.password_buffer):
            self.visible = True
            self.auth_pending = False
            self.password_buffer = ""
            self.auth_error = ""
            self._auth_buttons = []
            self.set_message("SERVICE MODE UNLOCKED", 1.5)
            return

        self.auth_error = "PASSWORD ERROR"
        self.auth_error_until = time.time() + 2.0
        self.password_buffer = ""

    def handle_key(self, key: int) -> bool:
        if not self.auth_pending:
            return False

        key = _normalize_cv_key(key)

        if key in (-1, 255):
            return True
        if key == 27:
            self._cancel_auth()
            return True
        if key in (10, 13):
            self._submit_password()
            return True
        if key in (8, 127):
            self.password_buffer = self.password_buffer[:-1]
            return True

        if 32 <= key <= 126 and len(self.password_buffer) < 32:
            self.password_buffer += chr(key)
        return True

    def handle_mouse(self, event: int, x: int, y: int) -> bool:
        if self.auth_pending:
            if event != cv2.EVENT_LBUTTONDOWN:
                return True

            for button in self._auth_buttons:
                x1, y1, x2, y2 = button.get("rect", (0, 0, 0, 0))
                if x1 <= x <= x2 and y1 <= y <= y2:
                    action = str(button.get("action", ""))
                    if action == "auth_open":
                        self._submit_password()
                    elif action == "auth_cancel":
                        self._cancel_auth()
                    elif action == "auth_backspace":
                        self.password_buffer = self.password_buffer[:-1]
                    elif action == "auth_clear":
                        self.password_buffer = ""
                    return True
            return True
        if not self.visible:
            return False

        panel_rect = self._panel_rect
        if panel_rect is not None:
            px1, py1, px2, py2 = panel_rect
            if not (px1 <= x <= px2 and py1 <= y <= py2):
                return False

        if event != cv2.EVENT_LBUTTONDOWN:
            return True

        for button in self._buttons:
            x1, y1, x2, y2 = button["rect"]
            if x1 <= x <= x2 and y1 <= y <= y2:
                if button.get("enabled", True):
                    action = str(button.get("action", ""))
                    if action == "close":
                        self.close()
                    elif action:
                        self._action = action
                return True
        return True

    def _draw_auth_modal(self, img: np.ndarray):
        h, w = img.shape[:2]
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.62, img, 0.38, 0, img)

        box_w = min(700, max(460, int(w * 0.42)))
        box_h = 300
        x1 = (w - box_w) // 2
        y1 = (h - box_h) // 2
        x2 = x1 + box_w
        y2 = y1 + box_h
        cv2.rectangle(img, (x1, y1), (x2, y2), (24, 24, 24), -1)
        cv2.rectangle(img, (x1, y1), (x2, y2), (0, 190, 255), 2)
        _put_text(img, "SERVICE / PLC DIAGNOSTIC", x1 + 24, y1 + 42, 0.68, (0, 220, 255), 1)
        _put_text(img, "Password (English keyboard)", x1 + 24, y1 + 88, 0.48, (210, 210, 210), 1)
        cv2.rectangle(img, (x1 + 24, y1 + 104), (x2 - 24, y1 + 152), (52, 52, 52), -1)
        cv2.rectangle(img, (x1 + 24, y1 + 104), (x2 - 24, y1 + 152), (160, 160, 160), 1)
        masked = "*" * len(self.password_buffer)
        _put_text(img, masked, x1 + 38, y1 + 137, 0.72, (255, 255, 255), 1)

        button_y1 = y1 + 178
        button_y2 = button_y1 + 42
        gap = 10
        button_w = max(96, (box_w - 48 - gap * 3) // 4)
        labels = [
            ("BACK", "auth_backspace", (80, 80, 80)),
            ("CLEAR", "auth_clear", (80, 80, 80)),
            ("CANCEL", "auth_cancel", (70, 70, 110)),
            ("OPEN", "auth_open", (0, 120, 0)),
        ]
        self._auth_buttons = []
        bx = x1 + 24
        for label, action, accent in labels:
            rect = (bx, button_y1, bx + button_w, button_y2)
            _draw_button(img, rect, label, True, accent)
            self._auth_buttons.append({"rect": rect, "action": action})
            bx += button_w + gap

        _put_text(
            img,
            "Type password, then ENTER or click OPEN   |   ESC: cancel",
            x1 + 24,
            y2 - 30,
            0.41,
            (180, 180, 180),
            1,
        )
        if self.auth_error and time.time() <= self.auth_error_until:
            _put_text(img, self.auth_error, x2 - 196, y1 + 88, 0.46, (0, 0, 255), 1)

    def draw(
        self,
        img: np.ndarray,
        plc_snapshot: Optional[Dict[str, Any]] = None,
        recent_events: Optional[List[Dict[str, Any]]] = None,
        app_state: Any = None,
        roi_debug: Optional[np.ndarray] = None,
        latest_log_path: str = "",
        error_log_path: str = "",
        test_summary: Optional[Dict[str, Any]] = None,
        soak_summary: Optional[Dict[str, Any]] = None,
    ) -> np.ndarray:
        if app_state is not None and bool(getattr(app_state, "edit_mode", False)):
            self.close()
            return img
        if self.auth_pending:
            self._draw_auth_modal(img)
            return img
        if not self.visible:
            return img

        plc_snapshot = plc_snapshot or {}
        recent_events = recent_events or []
        test_summary = test_summary or {}
        soak_summary = soak_summary or {}
        h, w = img.shape[:2]
        pw = min(self.panel_width, max(380, w - 320))
        x0 = w - pw
        self._panel_rect = (x0, 0, w - 1, h - 1)

        # Blend only the right-side panel region. The previous full-frame
        # img.copy()/addWeighted path copied the whole 1920x1080 image every
        # frame while SERVICE was open.
        panel_roi = img[:, x0:w]
        if self.opacity >= 0.95:
            panel_roi[:] = (18, 22, 28)
        else:
            background = np.empty_like(panel_roi)
            background[:] = (18, 22, 28)
            cv2.addWeighted(
                background,
                self.opacity,
                panel_roi,
                1.0 - self.opacity,
                0,
                panel_roi,
            )
        cv2.line(img, (x0, 0), (x0, h), (0, 200, 255), 2)

        self._buttons = []
        pad = 16
        cx = x0 + pad
        right = w - pad
        y = 30

        _put_text(img, "SERVICE / PLC DIAGNOSTIC", cx, y, 0.62, (0, 220, 255), 1)
        close_rect = (right - 72, 8, right, 38)
        _draw_button(img, close_rect, "CLOSE", True, (70, 70, 70))
        self._buttons.append({"rect": close_rect, "action": "close", "enabled": True})

        serial_open = bool(plc_snapshot.get("serial_open", False))
        comm_fault = bool(plc_snapshot.get("comm_fault_active", False))
        line_color = (0, 210, 0) if serial_open and not comm_fault else (0, 0, 255)
        y += 35
        _put_text(
            img,
            f"LINK {'OPEN' if serial_open else 'CLOSED'}   COMM {'FAULT' if comm_fault else 'OK'}   "
            f"SLAVE {plc_snapshot.get('slave_id', '-')}   "
            f"RX#{_safe_int(plc_snapshot.get('rx_count'))} TX#{_safe_int(plc_snapshot.get('tx_count'))}",
            cx,
            y,
            0.47,
            line_color,
            1,
        )

        y += 28
        heartbeat_epoch = plc_snapshot.get("heartbeat_last_change_epoch")
        try:
            heartbeat_moving = (
                heartbeat_epoch is not None
                and (time.time() - float(heartbeat_epoch)) <= 2.0
            )
        except Exception:
            heartbeat_moving = False

        regs = [
            ("D200 CMD", _safe_int(plc_snapshot.get("command")), CMD_NAMES),
            ("D201 STATUS", _safe_int(plc_snapshot.get("status")), STATUS_NAMES),
            ("D202 RESULT", _safe_int(plc_snapshot.get("result")), RESULT_NAMES),
            (
                "D203 HEART",
                _safe_int(plc_snapshot.get("heartbeat")),
                { _safe_int(plc_snapshot.get("heartbeat")): "MOVING" if heartbeat_moving else "STOPPED" },
            ),
        ]
        for label, value, names in regs:
            if label.startswith("D201") and value == 3:
                color = (0, 0, 255)
            elif label.startswith("D202") and value == 1:
                color = (0, 220, 0)
            elif label.startswith("D202") and value == 2:
                color = (0, 0, 255)
            else:
                color = (225, 225, 225)
            suffix = names.get(value, "") if names else ""
            _put_text(img, f"{label:<12} {value:5d}  {suffix}", cx, y, 0.49, color, 1)
            y += 24

        y += 2
        rx_summary = _ellipsize(plc_snapshot.get("last_rx_summary", "-"), 35)
        tx_summary = _ellipsize(plc_snapshot.get("last_tx_summary", "-"), 35)
        rx_hex = _ellipsize(plc_snapshot.get("last_rx_hex", ""), 28)
        tx_hex = _ellipsize(plc_snapshot.get("last_tx_hex", ""), 28)
        _put_text(
            img,
            f"RX {_age_text(plc_snapshot.get('last_rx_epoch'))}: {rx_summary} | {rx_hex}",
            cx,
            y,
            0.36,
            (90, 220, 255),
            1,
        )
        y += 19
        _put_text(
            img,
            f"TX {_age_text(plc_snapshot.get('last_tx_epoch'))}: {tx_summary} | {tx_hex}",
            cx,
            y,
            0.36,
            (255, 190, 80),
            1,
        )
        y += 25

        error_code = _safe_int(plc_snapshot.get("error_code"))
        error_detail = str(plc_snapshot.get("error_detail", "") or "")
        error_active = _safe_int(plc_snapshot.get("status")) == 3 or error_code != 0
        box_h = 72 if error_active else 48
        cv2.rectangle(img, (cx, y), (right, y + box_h), (35, 35, 35), -1)
        cv2.rectangle(img, (cx, y), (right, y + box_h), (0, 0, 255) if error_active else (80, 130, 80), 2)
        if error_active:
            _put_text(img, f"ERROR ACTIVE  {error_code} {ERROR_NAMES.get(error_code, 'UNDEFINED')}", cx + 10, y + 24, 0.48, (0, 0, 255), 1)
            _put_text(img, _ellipsize(error_detail, 62), cx + 10, y + 48, 0.38, (220, 220, 220), 1)
            reset_text = (
                "PLC RESET OR LOCAL RECOVER"
                if bool(getattr(app_state, "test_error_active", False))
                else "PLC D200=3 RESET REQUIRED"
            )
            _put_text(img, reset_text, cx + 10, y + 66, 0.38, (0, 200, 255), 1)
        else:
            _put_text(img, "NO ACTIVE ERROR", cx + 10, y + 29, 0.48, (0, 210, 0), 1)
        y += box_h + 12

        test_enabled = bool(test_summary.get("enabled", False))
        command_value = _safe_int(plc_snapshot.get("command"))
        status_value = _safe_int(plc_snapshot.get("status"))
        active_error_code = _safe_int(plc_snapshot.get("error_code"))
        service_test_active = bool(getattr(app_state, "test_error_active", False))
        edit_mode = bool(getattr(app_state, "edit_mode", False))
        soak_test_running = bool(soak_summary.get("active", False))
        real_error_active = error_active and not service_test_active

        ready_common = (
            test_enabled
            and not edit_mode
            and serial_open
            and not comm_fault
            and not service_test_active
            and not soak_test_running
            and not real_error_active
            and status_value != 8
            and command_value not in (8, 9)
        )
        logic_ready = ready_common
        recovery_ready = ready_common
        camera_recovery_ready = bool(
            test_enabled
            and bool(test_summary.get("camera_recovery_capable", False))
            and not edit_mode
            and not service_test_active
            and not soak_test_running
            and not real_error_active
            and status_value != 8
            and command_value not in (8, 9)
        )

        if not test_enabled:
            block_reason = "TEST DISABLED"
        elif edit_mode:
            block_reason = "TEST AVAILABLE IN RUN MODE ONLY"
        elif service_test_active:
            block_reason = "WAIT PLC RESET OR USE LOCAL RECOVER"
        elif soak_test_running:
            block_reason = "SOAK TEST ACTIVE - FAULT TEST BUTTONS LOCKED"
        elif real_error_active:
            block_reason = "BLOCKED: REAL ERROR IS ACTIVE"
        elif status_value == 8 or command_value in (8, 9):
            block_reason = "BLOCKED: SHUTDOWN / EMERGENCY"
        elif not serial_open or comm_fault:
            block_reason = "BLOCKED: PLC LINK NOT READY"
        else:
            block_reason = f"READY FROM CURRENT STATE D200={command_value} D201={status_value}"

        labels = [
            ("INSPECT", "inspection"),
            ("LIGHT", "light"),
            ("CAMERA", "camera"),
            ("PLC COMM", "plc_comm"),
        ]
        btn_gap = 6
        btn_h = 34
        btn_w = max(72, (right - cx - btn_gap * 3) // 4)

        _put_text(img, "PLC LOGIC TEST", cx, y, 0.46, (0, 220, 255), 1)
        y += 8
        for idx, (label, error_type) in enumerate(labels):
            x1 = cx + idx * (btn_w + btn_gap)
            y1 = y + 8
            rect = (x1, y1, x1 + btn_w, y1 + btn_h)
            _draw_button(img, rect, label, logic_ready, (70, 95, 145))
            self._buttons.append({
                "rect": rect,
                "action": f"test:logic:{error_type}",
                "enabled": logic_ready,
            })
        y += btn_h + 22

        _put_text(img, "SAFE RECOVERY TEST", cx, y, 0.46, (0, 220, 255), 1)
        y += 8
        for idx, (label, error_type) in enumerate(labels):
            x1 = cx + idx * (btn_w + btn_gap)
            y1 = y + 8
            rect = (x1, y1, x1 + btn_w, y1 + btn_h)
            button_ready = (
                camera_recovery_ready if error_type == "camera" else recovery_ready
            )
            _draw_button(img, rect, label, button_ready, (90, 105, 155))
            self._buttons.append({
                "rect": rect,
                "action": f"test:recovery:{error_type}",
                "enabled": button_ready,
            })
        y += btn_h + 16

        local_rect = (cx, y, right, y + 32)
        _draw_button(
            img,
            local_rect,
            "LOCAL RECOVER (TEST ONLY)",
            service_test_active,
            (95, 80, 130),
        )
        self._buttons.append({
            "rect": local_rect,
            "action": "test:local_recover",
            "enabled": service_test_active,
        })
        y += 44

        soak_enabled = bool(soak_summary.get("enabled", False))
        soak_active = bool(soak_summary.get("active", False))
        soak_phase = str(soak_summary.get("phase", "IDLE") or "IDLE")
        soak_cycles = _safe_int(soak_summary.get("cycle_count"))
        soak_ok = _safe_int(soak_summary.get("ok_count"))
        soak_ng = _safe_int(soak_summary.get("ng_count"))
        soak_errors = _safe_int(soak_summary.get("error_count"))
        soak_next = float(soak_summary.get("next_in_sec", 0.0) or 0.0)
        soak_log = os.path.basename(str(soak_summary.get("log_path", "") or ""))
        soak_start_ready = (
            soak_enabled
            and not soak_active
            and ready_common
            and status_value != 1
        )

        _put_text(
            img,
            "OVERNIGHT SOAK TEST (LOCAL 30s CYCLE)",
            cx,
            y,
            0.43,
            (0, 220, 255),
            1,
        )
        y += 8
        soak_gap = 8
        soak_w = max(120, (right - cx - soak_gap) // 2)
        start_rect = (cx, y + 8, cx + soak_w, y + 42)
        stop_rect = (cx + soak_w + soak_gap, y + 8, right, y + 42)
        _draw_button(img, start_rect, "START AUTO CYCLE", soak_start_ready, (50, 120, 70))
        _draw_button(img, stop_rect, "STOP / FINALIZE LOG", soak_active, (70, 70, 140))
        self._buttons.append({
            "rect": start_rect,
            "action": "soak:start",
            "enabled": soak_start_ready,
        })
        self._buttons.append({
            "rect": stop_rect,
            "action": "soak:stop",
            "enabled": soak_active,
        })
        y += 52
        soak_color = (0, 210, 0) if soak_active else (175, 175, 175)
        _put_text(
            img,
            f"SOAK {soak_phase}  CYCLE {soak_cycles}  OK {soak_ok}  NG {soak_ng}  ERR {soak_errors}  NEXT {soak_next:.1f}s",
            cx,
            y,
            0.34,
            soak_color,
            1,
        )
        y += 17
        _put_text(
            img,
            f"SOAK LOG: {_ellipsize(soak_log, 44)}",
            cx,
            y,
            0.31,
            (165, 165, 165),
            1,
        )
        y += 20

        mode = str(test_summary.get("mode", "") or "-").upper()
        phase = str(test_summary.get("phase", "IDLE") or "IDLE")
        test_type = str(test_summary.get("type", "") or "-").upper()
        test_result = str(test_summary.get("result", "") or "-").upper()
        expected_code = _safe_int(test_summary.get("expected_code"))
        actual_code = _safe_int(test_summary.get("actual_code"))
        health_detail = _ellipsize(test_summary.get("health_detail", ""), 58)
        log_ok = bool(test_summary.get("log_saved", False))
        pre_command = _safe_int(test_summary.get("pre_command"))
        pre_status = _safe_int(test_summary.get("pre_status"))
        pre_result = _safe_int(test_summary.get("pre_result"))
        reset_source = str(test_summary.get("reset_source", "") or "-")
        recovery_ms = _safe_int(test_summary.get("recovery_elapsed_ms"))

        status_color = (
            (0, 210, 0)
            if test_result == "PASS"
            else ((0, 0, 255) if test_result == "FAIL" else (210, 210, 210))
        )
        cv2.rectangle(img, (cx, y), (right, y + 94), (34, 34, 34), -1)
        cv2.rectangle(img, (cx, y), (right, y + 94), status_color, 1)
        _put_text(
            img,
            f"MODE {mode}   TEST {test_type}   PHASE {phase}   RESULT {test_result}",
            cx + 8,
            y + 20,
            0.38,
            status_color,
            1,
        )
        _put_text(
            img,
            f"PRE C/S/R={pre_command}/{pre_status}/{pre_result}   RESET={reset_source}   {recovery_ms}ms",
            cx + 8,
            y + 40,
            0.34,
            (200, 200, 200),
            1,
        )
        _put_text(
            img,
            f"EXPECTED E{expected_code:02d}   ACTUAL E{actual_code:02d}   LOG {'SAVED' if log_ok else '-'}",
            cx + 8,
            y + 61,
            0.35,
            (210, 210, 210),
            1,
        )
        _put_text(
            img,
            health_detail or block_reason,
            cx + 8,
            y + 83,
            0.33,
            (0, 210, 255) if ready_common else (150, 190, 255),
            1,
        )
        y += 104

        _put_text(img, block_reason, cx, y, 0.36, (160, 200, 255), 1)
        y += 18
        _put_text(
            img,
            "HARD TEST NOT INCLUDED - PHYSICAL FAULTS ARE LOGGED ONLY",
            cx,
            y,
            0.32,
            (145, 145, 145),
            1,
        )
        y += 20

        _put_text(
            img,
            f"TEST LOG: {_ellipsize(os.path.basename(latest_log_path), 36)}",
            cx,
            y,
            0.33,
            (180, 180, 180),
            1,
        )
        y += 17
        _put_text(
            img,
            f"ERROR LOG: {_ellipsize(os.path.basename(error_log_path), 35)}",
            cx,
            y,
            0.33,
            (180, 180, 180),
            1,
        )
        y += 23

        _put_text(img, "RECENT RX / TX / STATE", cx, y, 0.46, (0, 220, 255), 1)
        y += 20
        shown = 0
        for event in reversed(recent_events):
            direction = str(event.get("direction", ""))
            if direction not in ("RX", "TX", "ERROR", "STATE", "APP", "SYSTEM"):
                continue
            if direction == "STATE" and _safe_int(event.get("register"), -1) == 203:
                # Heartbeat 변화는 상단 D203에서 별도 표시하므로 최근 통신 목록에서는 제외한다.
                continue
            summary = _ellipsize(event.get("summary", ""), 52)
            clock = str(event.get("timestamp", ""))
            clock = clock[11:19] if len(clock) >= 19 else clock
            color = {
                "RX": (90, 220, 255),
                "TX": (255, 190, 80),
                "ERROR": (0, 0, 255),
                "STATE": (200, 200, 200),
                "APP": (180, 160, 255),
                "SYSTEM": (170, 170, 170),
            }.get(direction, (190, 190, 190))
            _put_text(img, f"{clock} {direction:<5} {summary}", cx, y, 0.34, color, 1)
            y += 18
            shown += 1
            if shown >= self.rx_tx_rows:
                break

        available_h = h - y - 18
        if self.show_roi_debug and roi_debug is not None and roi_debug.size > 0 and available_h >= 100:
            y += 6
            _put_text(img, "ROI INSPECTION DEBUG", cx, y, 0.46, (0, 220, 255), 1)
            y += 10
            target_w = right - cx
            target_h = max(80, h - y - 10)
            rh, rw = roi_debug.shape[:2]
            scale = min(target_w / max(1, rw), target_h / max(1, rh))
            nw = max(1, int(rw * scale))
            nh = max(1, int(rh * scale))
            cache_key = (id(roi_debug), rw, rh, nw, nh)
            if cache_key != self._roi_resize_cache_key:
                self._roi_resize_cache = cv2.resize(
                    roi_debug,
                    (nw, nh),
                    interpolation=cv2.INTER_AREA,
                )
                self._roi_resize_cache_key = cache_key
            resized = self._roi_resize_cache
            x_img = cx + (target_w - nw) // 2
            y_img = y + max(0, (target_h - nh) // 2)
            cv2.rectangle(img, (cx, y), (right, h - 8), (45, 45, 45), -1)
            img[y_img:y_img + nh, x_img:x_img + nw] = resized
            cv2.rectangle(img, (cx, y), (right, h - 8), (100, 100, 100), 1)

        if self._message and time.time() <= self._message_until:
            msg_w = min(pw - 32, 500)
            mx1 = x0 + (pw - msg_w) // 2
            my2 = h - 12
            my1 = my2 - 42
            cv2.rectangle(img, (mx1, my1), (mx1 + msg_w, my2), (25, 25, 25), -1)
            cv2.rectangle(img, (mx1, my1), (mx1 + msg_w, my2), (0, 210, 255), 1)
            _put_text(img, _ellipsize(self._message, 58), mx1 + 10, my1 + 27, 0.40, (255, 255, 255), 1)

        return img
