from pyzbar import pyzbar
import cv2
import re


def _normalize_barcode_text(s: str) -> str:
    s = str(s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s


def run_barcode(crop, cfg):
    base = crop
    if base is None or base.size == 0:
        return False, {"barcode_detected": False}, "BARCODE_EMPTY"

    decode_candidates = [
        ("raw", base),
        ("up2", cv2.resize(base, None, fx=2, fy=2, interpolation=cv2.INTER_CUBIC)),
        ("up3", cv2.resize(base, None, fx=3, fy=3, interpolation=cv2.INTER_CUBIC)),
    ]

    allowed_types = cfg.get("allowed_types") or []
    allowed_types = [str(x).upper() for x in allowed_types]

    for _name, img in decode_candidates:
        try:
            zres = pyzbar.decode(img)
        except Exception:
            zres = []

        for z in zres:
            z_type = str(getattr(z, "type", "") or "").upper()
            z_text = z.data.decode("utf-8", errors="ignore")

            if allowed_types and z_type not in allowed_types:
                continue

            return True, {
                "barcode_detected": True,
                "barcode_type": z_type,
                "barcode_text": z_text,
                "barcode_text_norm": _normalize_barcode_text(z_text),
                "_last_image": base,
            }, "OK"

    return False, {
        "barcode_detected": False,
        "_last_image": base,
    }, "NG: BARCODE SCAN FAIL"


def eval_barcode(ok, metrics, reason, cfg, recipe_default, runtime_cfg):

    print(
        "[DBG BARCODE]",
        "det=", metrics.get("barcode_detected"),
        "type=", metrics.get("barcode_type"),
        "text=", repr(metrics.get("barcode_text")),
        "norm=", repr(metrics.get("barcode_text_norm")),
    )
    if not metrics.get("barcode_detected"):
        return False, "NG: BARCODE SCAN FAIL"

    digits_only = bool(cfg.get("digits_only", False))
    exact_length = cfg.get("exact_length", None)

    # 숫자만 체크
    if digits_only:
        if not text_norm.isdigit():
            return False, "NG: BARCODE NOT DIGITS"

    # 길이 고정 체크
    if exact_length is not None:
        if len(text_norm) != int(exact_length):
            return False, "NG: BARCODE LENGTH MISMATCH"

    expected_text = str(cfg.get("expected_text", "") or "").strip()
    expected_prefix = str(cfg.get("expected_prefix", "") or "").strip()
    min_length = int(cfg.get("min_length", 1))

    if not text:
        return False, "NG: BARCODE EMPTY"

    if len(text_norm) < min_length:
        return False, "NG: BARCODE TOO SHORT"

    if expected_text:
        if text_norm != _normalize_barcode_text(expected_text):
            return False, "NG: BARCODE INVALID"

    if expected_prefix:
        if not text_norm.startswith(_normalize_barcode_text(expected_prefix)):
            return False, "NG: BARCODE INVALID"
        
    
    # OCR cross-check
    compare_ocr = bool(cfg.get("compare_ocr", False))

    if compare_ocr:
        ocr_text = str(metrics.get("ocr_text_norm", "") or "")
        if ocr_text:
            if text_norm != ocr_text:
                return False, "NG: BARCODE OCR MISMATCH"

    return True, f"OK: {text}"