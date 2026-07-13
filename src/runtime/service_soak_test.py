import json
import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, Optional


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(k): _json_safe(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(v) for v in value]
    return str(value)


class ServiceSoakTest:
    """Small, test-only overnight cycle logger/state holder.

    The class never opens camera, light or PLC devices.  It only stores the
    state of a service soak session and writes a dedicated JSONL/summary log.
    Production command handling remains in main_vp.py.
    """

    def __init__(self, cfg: Optional[Dict[str, Any]], logs_root: str):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_sec = max(5.0, float(cfg.get("interval_sec", 30.0)))
        self.reset_delay_sec = max(0.2, float(cfg.get("reset_delay_sec", 2.0)))
        self.start_delay_sec = max(0.2, float(cfg.get("start_delay_sec", 1.0)))
        self.keep_days = max(1, int(cfg.get("keep_days", 14)))
        self.keep_sessions = max(2, int(cfg.get("keep_sessions", 20)))

        rel_dir = str(cfg.get("log_dir", "soak_tests") or "soak_tests").strip()
        self.log_root = (
            os.path.abspath(rel_dir)
            if os.path.isabs(rel_dir)
            else os.path.abspath(os.path.join(logs_root, rel_dir))
        )

        self.active = False
        self.phase = "IDLE"
        self.session_id = ""
        self.started_epoch = 0.0
        self.stopped_epoch = 0.0
        self.next_action_epoch = 0.0
        self.last_cycle_started_epoch = 0.0
        self.last_cycle_finished_epoch = 0.0
        self.cycle_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.error_count = 0
        self.last_result = ""
        self.last_elapsed_ms = 0
        self.last_message = ""
        self.stop_reason = ""
        self.log_path = ""
        self.summary_path = ""
        self._fp = None

    def _now_text(self) -> str:
        return datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]

    def _write(self, event: str, payload: Optional[Dict[str, Any]] = None):
        if self._fp is None:
            return
        record = {
            "schema_version": 1,
            "timestamp": self._now_text(),
            "epoch": time.time(),
            "session_id": self.session_id,
            "event": str(event),
            "phase": str(self.phase),
            "cycle": int(self.cycle_count),
            "counts": {
                "ok": int(self.ok_count),
                "ng": int(self.ng_count),
                "error": int(self.error_count),
            },
        }
        if payload:
            record.update(_json_safe(payload))
        self._fp.write(json.dumps(record, ensure_ascii=False, separators=(",", ":")) + "\n")
        self._fp.flush()

    def _prune(self):
        try:
            if not os.path.isdir(self.log_root):
                return
            cutoff = time.time() - (self.keep_days * 86400.0)
            sessions = []
            for name in os.listdir(self.log_root):
                path = os.path.join(self.log_root, name)
                if not os.path.isdir(path):
                    continue
                try:
                    mtime = os.path.getmtime(path)
                except OSError:
                    continue
                sessions.append((mtime, path))
            sessions.sort(reverse=True)
            for index, (mtime, path) in enumerate(sessions):
                if index < self.keep_sessions and mtime >= cutoff:
                    continue
                for root, dirs, files in os.walk(path, topdown=False):
                    for filename in files:
                        try:
                            os.remove(os.path.join(root, filename))
                        except OSError:
                            pass
                    for dirname in dirs:
                        try:
                            os.rmdir(os.path.join(root, dirname))
                        except OSError:
                            pass
                try:
                    os.rmdir(path)
                except OSError:
                    pass
        except Exception as e:
            print(f"[SOAK TEST] prune failed: {e}")

    def start(
        self,
        plc_snapshot: Dict[str, Any],
        vision_snapshot: Dict[str, Any],
    ) -> bool:
        if not self.enabled or self.active:
            return False

        now = time.time()
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.session_id = f"{stamp}_{uuid.uuid4().hex[:8]}"
        session_dir = os.path.join(self.log_root, self.session_id)
        os.makedirs(session_dir, exist_ok=True)
        self.log_path = os.path.join(session_dir, "soak_cycle.jsonl")
        self.summary_path = os.path.join(session_dir, "soak_summary.json")
        self._fp = open(self.log_path, "a", encoding="utf-8", buffering=1)

        self.active = True
        self.phase = "WAIT_INSPECT"
        self.started_epoch = now
        self.stopped_epoch = 0.0
        self.next_action_epoch = now + self.start_delay_sec
        self.last_cycle_started_epoch = 0.0
        self.last_cycle_finished_epoch = 0.0
        self.cycle_count = 0
        self.ok_count = 0
        self.ng_count = 0
        self.error_count = 0
        self.last_result = ""
        self.last_elapsed_ms = 0
        self.last_message = "STARTED"
        self.stop_reason = ""

        self._write(
            "SESSION_START",
            {
                "settings": {
                    "interval_sec": self.interval_sec,
                    "reset_delay_sec": self.reset_delay_sec,
                    "start_delay_sec": self.start_delay_sec,
                    "source": "SERVICE_LOCAL_COMMAND_PATH",
                },
                "plc_state": plc_snapshot,
                "vision_state": vision_snapshot,
            },
        )
        self._prune()
        return True

    def begin_cycle(self, plc_snapshot: Dict[str, Any], health: Dict[str, Any]):
        self.cycle_count += 1
        self.phase = "INSPECTING"
        self.last_cycle_started_epoch = time.time()
        self.last_message = f"CYCLE {self.cycle_count} INSPECTING"
        self._write(
            "CYCLE_START",
            {
                "plc_state": plc_snapshot,
                "health": health,
            },
        )

    def complete_inspection(
        self,
        overall_ok: bool,
        elapsed_ms: int,
        plc_snapshot: Dict[str, Any],
        inspection_summary: Dict[str, Any],
        health: Dict[str, Any],
    ):
        self.last_result = "OK" if overall_ok else "NG"
        self.last_elapsed_ms = int(elapsed_ms)
        if overall_ok:
            self.ok_count += 1
        else:
            self.ng_count += 1
        self.phase = "WAIT_RESET"
        self.next_action_epoch = time.time() + self.reset_delay_sec
        self.last_message = (
            f"CYCLE {self.cycle_count} {self.last_result} - RESET IN "
            f"{self.reset_delay_sec:.1f}s"
        )
        self._write(
            "INSPECTION_RESULT",
            {
                "result": self.last_result,
                "elapsed_ms": self.last_elapsed_ms,
                "plc_state": plc_snapshot,
                "inspection": inspection_summary,
                "health": health,
            },
        )

    def complete_reset(
        self,
        plc_snapshot: Dict[str, Any],
        health: Dict[str, Any],
        source: str = "SERVICE_LOCAL",
    ):
        self.phase = "WAIT_INSPECT"
        self.last_cycle_finished_epoch = time.time()
        target = self.last_cycle_started_epoch + self.interval_sec
        self.next_action_epoch = max(time.time() + 0.1, target)
        self.last_message = (
            f"CYCLE {self.cycle_count} RESET {source} - "
            f"NEXT IN {max(0.0, self.next_action_epoch - time.time()):.1f}s"
        )
        self._write(
            "RESET_COMPLETE",
            {
                "plc_state": plc_snapshot,
                "health": health,
                "reset_source": str(source),
                "next_cycle_epoch": self.next_action_epoch,
            },
        )

    def defer(self, reason: str, seconds: float = 0.5):
        self.last_message = str(reason)
        self.next_action_epoch = time.time() + max(0.1, float(seconds))

    def record_external_event(self, event: str, payload: Optional[Dict[str, Any]] = None):
        self._write(str(event), payload or {})

    def fail(
        self,
        reason: str,
        plc_snapshot: Dict[str, Any],
        vision_snapshot: Dict[str, Any],
        exception: Optional[BaseException] = None,
    ):
        self.error_count += 1
        self.phase = "ERROR"
        self.last_result = "ERROR"
        self.last_message = str(reason)
        payload = {
            "reason": str(reason),
            "plc_state": plc_snapshot,
            "vision_state": vision_snapshot,
        }
        if exception is not None:
            payload["exception"] = {
                "type": type(exception).__name__,
                "message": str(exception),
                "repr": repr(exception),
            }
        self._write("SESSION_ERROR", payload)
        self.stop("ERROR_STOP", plc_snapshot, vision_snapshot)

    def stop(
        self,
        reason: str,
        plc_snapshot: Optional[Dict[str, Any]] = None,
        vision_snapshot: Optional[Dict[str, Any]] = None,
    ) -> bool:
        if not self.active and self._fp is None:
            return False

        self.stop_reason = str(reason or "STOPPED")
        self.stopped_epoch = time.time()
        previous_phase = self.phase
        self.phase = "STOPPED"
        self.last_message = self.stop_reason
        self._write(
            "SESSION_STOP",
            {
                "reason": self.stop_reason,
                "previous_phase": previous_phase,
                "duration_sec": max(0.0, self.stopped_epoch - self.started_epoch),
                "plc_state": plc_snapshot or {},
                "vision_state": vision_snapshot or {},
            },
        )

        if self._fp is not None:
            try:
                self._fp.close()
            except Exception:
                pass
            self._fp = None

        summary = self.summary()
        try:
            temp_path = self.summary_path + ".tmp"
            with open(temp_path, "w", encoding="utf-8") as file:
                json.dump(_json_safe(summary), file, ensure_ascii=False, indent=2)
            os.replace(temp_path, self.summary_path)
        except Exception as e:
            print(f"[SOAK TEST] summary save failed: {e}")

        self.active = False
        return True

    def summary(self) -> Dict[str, Any]:
        now = time.time()
        remaining = 0.0
        if self.active and self.next_action_epoch > 0:
            remaining = max(0.0, self.next_action_epoch - now)
        return {
            "enabled": bool(self.enabled),
            "active": bool(self.active),
            "phase": str(self.phase),
            "session_id": str(self.session_id),
            "started_at": (
                datetime.fromtimestamp(self.started_epoch).strftime("%Y-%m-%d %H:%M:%S")
                if self.started_epoch > 0 else ""
            ),
            "duration_sec": max(
                0.0,
                (self.stopped_epoch or now) - self.started_epoch,
            ) if self.started_epoch > 0 else 0.0,
            "next_in_sec": remaining,
            "interval_sec": float(self.interval_sec),
            "cycle_count": int(self.cycle_count),
            "ok_count": int(self.ok_count),
            "ng_count": int(self.ng_count),
            "error_count": int(self.error_count),
            "last_result": str(self.last_result),
            "last_elapsed_ms": int(self.last_elapsed_ms),
            "last_message": str(self.last_message),
            "stop_reason": str(self.stop_reason),
            "log_path": str(self.log_path),
            "summary_path": str(self.summary_path),
            "source": "SERVICE_LOCAL_COMMAND_PATH",
        }
