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
    return bool(final_ok), ret_metrics, ("OK" if final_ok else last_reason)