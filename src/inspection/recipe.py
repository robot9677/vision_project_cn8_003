import json
from typing import Any, Dict

def _normalize_key(k: Any) -> str:
    s = str(k).strip()
    if s.upper().startswith("ROI"):
        return s.upper()
    if s.isdigit():
        return f"ROI{s}"
    return s.upper()

def load_recipe(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            raw = json.load(f)
    except FileNotFoundError:
        raw = {}

    default = raw.get("default", {"type": "mean_threshold", "min_mean": 0, "max_mean": 255})
    overrides_raw = raw.get("overrides", {})
    overrides = {_normalize_key(k): v for k, v in overrides_raw.items()}
    decision = raw.get("decision", {"mode": "any_fail_is_ng"})

    return {"default": default, "overrides": overrides, "decision": decision}

def get_roi_cfg(recipe: Dict[str, Any], roi_id: Any) -> Dict[str, Any]:
    base = dict(recipe.get("default", {}))
    overrides = recipe.get("overrides", {})

    rid_norm = _normalize_key(roi_id)
    if rid_norm in overrides:
        base.update(overrides[rid_norm])
    else:
        rid_str = str(roi_id).strip().upper()
        if rid_str in overrides:
            base.update(overrides[rid_str])

    return base

def save_recipe(path: str, recipe: Dict[str, Any]) -> None:
    import os, json
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)