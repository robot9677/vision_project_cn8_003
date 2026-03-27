from pyzbar import pyzbar
import cv2
import re
import pytesseract

def _normalize_qr_text(s: str) -> str:
    s = str(s or "").strip().upper()
    s = re.sub(r"[^A-Z0-9]", "", s)
    return s

def _ocr_bottom_text_from_qr_crop(crop):
    if crop is None or crop.size == 0:
        return ""

    h, w = crop.shape[:2]
    band = crop[int(h*0.72):h, 0:w]

    if band is None or band.size == 0:
        return ""

    band_up = cv2.resize(band, None, fx=3.0, fy=3.0, interpolation=cv2.INTER_CUBIC)
    band_blur = cv2.GaussianBlur(band_up, (3, 3), 0)
    _, band_bw = cv2.threshold(band_blur, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)

    config = "--psm 7 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789"

    try:
        txt = pytesseract.image_to_string(band_bw, config=config)
    except Exception:
        txt = ""

    return _normalize_qr_text(txt)


def run_qr(crop, cfg):
    base = crop
    if base is None or base.size == 0:
        return False, {"qr_detected": False}, "QR_EMPTY"

    decode_candidates = [
        ("raw", base),
        ("up2", cv2.resize(base, None, fx=2, fy=2)),
        ("up3", cv2.resize(base, None, fx=3, fy=3)),
    ]

    for name, img in decode_candidates:
        try:
            zres = pyzbar.decode(img)
        except:
            zres = []

        if zres:
            text = zres[0].data.decode("utf-8", errors="ignore")
            return True, {
                "qr_detected": True,
                "qr_text": text,
                "qr_text_norm": _normalize_qr_text(text),
                "qr_ocr_text": _ocr_bottom_text_from_qr_crop(base),
                "_last_image": base   
            }, "OK"

    return False, {"qr_detected": False, "_last_image": base}, "NG: QR SCAN FAIL"


def eval_qr(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    if not metrics.get("qr_detected"):
        return False, "NG: QR SCAN FAIL"

    if cfg.get("compare_ocr"):
        qr = str(metrics.get("qr_text_norm",""))
        ocr = str(metrics.get("qr_ocr_text",""))

        if not ocr:
            return False, "NG: OCR FAIL"

        # 완화 (prefix 비교)
        if not qr.endswith(ocr):
            return False, "NG: TEXT MISMATCH"

        return True, f"OK: {ocr}"

    return True, f"OK: {metrics.get('qr_text')}"