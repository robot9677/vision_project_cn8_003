# src/inspection/tools_enhance.py
import cv2
from typing import Any, Dict, Tuple
import numpy as np
from .toolchain import register_tool

def _noop(crop: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]) -> Tuple[np.ndarray, Dict[str, Any], bool, str]:
    return crop, {"enhance": "noop"}, True, "OK"

def _threshold(crop, params, ctx):
    mode = params.get("mode", "otsu")

    if len(crop.shape) == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    if mode == "otsu":
        th_val, th = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    else:
        th_val = float(params.get("th", 128))
        _, th = cv2.threshold(gray, th_val, 255, cv2.THRESH_BINARY)

    white_ratio = float((th > 0).sum()) / th.size

    meta = {
        "white_ratio": white_ratio,
        "th_value": float(th_val),
    }

    return th, meta, True, "OK"

def _clahe(crop, params, ctx):
    import cv2
    if crop is None or crop.size == 0:
        return crop, {}, False, "EMPTY"

    if crop.ndim == 3:
        gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    else:
        gray = crop

    clip = float(params.get("clip", 2.0))
    grid = int(params.get("grid", 8))
    clahe = cv2.createCLAHE(clipLimit=clip, tileGridSize=(grid, grid))
    out = clahe.apply(gray)
    return out, {"clahe": 1}, True, "OK"

def register_enhance_tools() -> None:

    register_tool("enhance.noop", _noop)
    register_tool("enhance.threshold", _threshold)   
    register_tool("enhance.clahe", _clahe)
    register_tool("enhance.gaussian_blur", _gaussian_blur)
    register_tool("enhance.blackhat", _blackhat)
    register_tool("enhance.adaptive_threshold", _adaptive_threshold)

def _gaussian_blur(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    if img is None or img.size == 0:
        return img, {}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    k = int(params.get("ksize", 5))
    if k < 1:
        k = 1
    if k % 2 == 0:
        k += 1

    sigma = float(params.get("sigma", 0))
    out = cv2.GaussianBlur(img8, (k, k), sigmaX=sigma)
    return out, {"blur_ksize": k, "blur_sigma": sigma}, True, "OK"


def _blackhat(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    if img is None or img.size == 0:
        return img, {}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    k = int(params.get("ksize", 15))
    if k < 3:
        k = 3
    if k % 2 == 0:
        k += 1

    shape_name = str(params.get("shape", "ellipse")).lower()
    shape = cv2.MORPH_ELLIPSE if shape_name == "ellipse" else cv2.MORPH_RECT
    ker = cv2.getStructuringElement(shape, (k, k))

    out = cv2.morphologyEx(img8, cv2.MORPH_BLACKHAT, ker)
    return out, {"blackhat_ksize": k}, True, "OK"

def _adaptive_threshold(img: np.ndarray, params: Dict[str, Any], ctx: Dict[str, Any]):
    if img is None or img.size == 0:
        return img, {}, False, "EMPTY"

    if img.dtype != np.uint8:
        img8 = cv2.normalize(img, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    else:
        img8 = img.copy()

    if img8.ndim == 3:
        img8 = cv2.cvtColor(img8, cv2.COLOR_BGR2GRAY)

    block_size = int(params.get("block_size", 31))
    if block_size < 3:
        block_size = 3
    if block_size % 2 == 0:
        block_size += 1

    c = float(params.get("C", -3))
    inv = bool(params.get("invert", False))

    mode = cv2.THRESH_BINARY_INV if inv else cv2.THRESH_BINARY
    out = cv2.adaptiveThreshold(
        img8,
        255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        mode,
        block_size,
        c,
    )

    white_ratio = float(np.count_nonzero(out)) / float(out.size) if out.size else 0.0
    return out, {
        "adaptive_block_size": block_size,
        "adaptive_C": c,
        "white_ratio": white_ratio,
    }, True, "OK"