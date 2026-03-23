import json
from copy import deepcopy
from typing import Any, Dict, List


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

    inspections_raw = raw.get("inspections", [])
    inspections: List[Dict[str, Any]] = []
    if isinstance(inspections_raw, list):
        for item in inspections_raw:
            if not isinstance(item, dict):
                continue
            row = deepcopy(item)
            row["roi_key"] = _normalize_key(row.get("roi_id", ""))
            row["enabled"] = bool(row.get("enabled", True))
            row["params"] = row.get("params", {}) if isinstance(row.get("params", {}), dict) else {}
            inspections.append(row)

    return {
        "default": default,
        "overrides": overrides,
        "decision": decision,
        "inspections": inspections,
    }


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


def get_inspection_cfgs(recipe: Dict[str, Any], roi_id: Any) -> List[Dict[str, Any]]:
    rid_norm = _normalize_key(roi_id)
    base = dict(recipe.get("default", {}))

    jobs = []
    for item in recipe.get("inspections", []):
        if not isinstance(item, dict):
            continue
        if not bool(item.get("enabled", True)):
            continue
        if item.get("roi_key") != rid_norm:
            continue

        params = item.get("params", {}) if isinstance(item.get("params", {}), dict) else {}
        cfg = deepcopy(base)
        cfg.update(params)
        cfg["id"] = item.get("id", f"{rid_norm}_job")
        cfg["name"] = item.get("id", f"{rid_norm}_job")
        cfg["type"] = str(item.get("type", cfg.get("type", ""))).strip()
        cfg["roi_id"] = roi_id
        jobs.append(cfg)

    if jobs:
        return jobs

    return [get_roi_cfg(recipe, roi_id)]


def save_recipe(path: str, recipe: Dict[str, Any]) -> None:
    import os

    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(recipe, f, ensure_ascii=False, indent=2)