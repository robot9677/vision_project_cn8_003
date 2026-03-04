from typing import Any, Dict, Tuple, Callable
import numpy as np
import cv2

AnalyzerFn = Callable[[np.ndarray, Dict[str, Any]], Tuple[bool, Dict[str, Any], str]]

def analyze_mean_threshold(crop: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    mean = float(np.mean(crop)) if crop.size else 0.0
    std  = float(np.std(crop)) if crop.size else 0.0

    mn = float(cfg.get("min_mean", 0))
    mx = float(cfg.get("max_mean", 255))

    ok = (mn <= mean <= mx)
    reason = "OK" if ok else ("LOW_MEAN" if mean < mn else "HIGH_MEAN")
    metrics = {"mean": mean, "std": std, "min_mean": mn, "max_mean": mx}
    return ok, metrics, reason

def analyze_edge_energy(crop: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    if crop.size == 0:
        return False, {"edge_var": 0.0}, "EMPTY_CROP"

    lap = cv2.Laplacian(crop, cv2.CV_64F)
    edge_var = float(lap.var())

    thr = float(cfg.get("min_edge", cfg.get("min_edge_var", 0.0)))
    ok = edge_var >= thr
    reason = "OK" if ok else "LOW_EDGE"
    metrics = {"edge_var": edge_var, "min_edge": thr}
    return ok, metrics, reason

# 타입 등록 테이블 (주요)
ANALYZERS: Dict[str, AnalyzerFn] = {
    # mean threshold
    "mean": analyze_mean_threshold,
    "mean_threshold": analyze_mean_threshold,
    "threshold": analyze_mean_threshold,
    "mean_score": analyze_mean_score,   
    
    # edge energy
    "edge": analyze_edge_energy,
    "edge_energy": analyze_edge_energy,
    "lap_var": analyze_edge_energy,
    "laplacian_var": analyze_edge_energy,
}

def run_analyzer(crop: np.ndarray, cfg: Dict[str, Any]) -> Tuple[bool, Dict[str, Any], str]:
    a_type = str(cfg.get("type", "mean_threshold")).strip().lower()
    fn = ANALYZERS.get(a_type)
    if fn is None:
        return False, {"error": f"unknown analyzer type: {a_type}"}, "UNKNOWN_ANALYZER"
    return fn(crop, cfg)

def analyze_mean_score(crop: np.ndarray, cfg: Dict[str, Any]):
    mean = float(np.mean(crop)) if crop.size else 0.0
    std  = float(np.std(crop)) if crop.size else 0.0

    mn = float(cfg.get("min_mean", 0))
    mx = float(cfg.get("max_mean", 255))
    min_score = float(cfg.get("min_score", 0.0))

    score = std

    ok_mean = mn <= mean <= mx
    ok_score = score >= min_score

    ok = ok_mean and ok_score

    if not ok_mean:
        reason = "MEAN_OUT"
    elif not ok_score:
        reason = "LOW_SCORE"
    else:
        reason = "OK"

    metrics = {
        "mean": mean,
        "score": score,
        "min_mean": mn,
        "max_mean": mx,
        "min_score": min_score
    }

    return ok, metrics, reason