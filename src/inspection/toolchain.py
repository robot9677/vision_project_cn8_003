# src/inspection/toolchain.py
from typing import Any, Dict, List, Tuple, Callable
import numpy as np

ToolFn = Callable[[np.ndarray, Dict[str, Any], Dict[str, Any]], Tuple[np.ndarray, Dict[str, Any], bool, str]]

_TOOL_REGISTRY: Dict[str, ToolFn] = {}

def register_tool(name: str, fn: ToolFn) -> None:
    _TOOL_REGISTRY[name] = fn

def run_toolchain(crop: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    """
    cfg:
      - tools: [{"tool":"enhance.noop", "params":{...}}, ...]
      - tool_decision: "all_ok"(default) | "any_ok" | "last"
    """
    steps: List[Dict[str, Any]] = cfg.get("tools") or []
    decision = (cfg.get("tool_decision") or "all_ok").strip().lower()

    ctx: Dict[str, Any] = {
        "metrics": {},
        "steps": [],
        "product_profile": cfg.get("product_profile")
    }
    
    cur = crop
    oks: List[bool] = []
    last_reason = "NO_TOOLS"

    for step in steps:
        name = str(step.get("tool", "")).strip()
        params = step.get("params") or {}
        fn = _TOOL_REGISTRY.get(name)

        if fn is None:
            out, meta, ok, reason = cur, {}, False, f"UNKNOWN_TOOL:{name}"
        else:
            out, meta, ok, reason = fn(cur, params, ctx)

        ctx["steps"].append({"tool": name, "ok": bool(ok), "reason": reason, "meta": meta})
        if meta:
            ctx["metrics"].update(meta)

        cur = out
        oks.append(bool(ok))
        last_reason = reason

    if not steps:
        return False, ctx["metrics"], "NO_TOOLS"

    if decision == "any_ok":
        final_ok = any(oks)
    elif decision == "last":
        final_ok = oks[-1]
    else:
        final_ok = all(oks)

    ctx["metrics"]["_last_image"] = cur
    ret_metrics = dict(ctx["metrics"])

    if final_ok:
        final_reason = "OK"

    elif decision == "last":
        # 마지막 Tool의 결과를 최종 판정으로 사용하는 경우
        final_reason = str(ctx["steps"][-1].get("reason") or "TOOL_FAILED")

    else:
        # all_ok 또는 any_ok 실패 시 실제로 실패한 첫 번째 Tool의 원인 유지
        failed_step = next(
            (step for step in ctx["steps"] if not bool(step.get("ok"))),
            None,
        )

        if failed_step is not None:
            final_reason = str(
                failed_step.get("reason") or
                f"TOOL_FAILED:{failed_step.get('tool', 'unknown')}"
            )
        else:
            final_reason = "TOOLCHAIN_FAILED"

    return bool(final_ok), ret_metrics, final_reason

def tool_measure_mean_raw_range(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    if crop is None or crop.size == 0:
        return crop, {"mean_raw": 0.0}, False, "EMPTY_CROP"

    mean_raw = float(np.mean(crop))

    min_mean_raw = params.get("min_mean_raw", None)
    max_mean_raw = params.get("max_mean_raw", None)

    ok = True
    reason = "OK"

    if min_mean_raw is not None and mean_raw < float(min_mean_raw):
        ok = False
        reason = "MEAN_RAW_LOW"

    if max_mean_raw is not None and mean_raw > float(max_mean_raw):
        ok = False
        reason = "MEAN_RAW_HIGH"

    meta = {
        "mean_raw": mean_raw,
        "min_mean_raw": min_mean_raw,
        "max_mean_raw": max_mean_raw,
    }

    return crop, meta, ok, reason


register_tool("measure.mean_raw_range", tool_measure_mean_raw_range)