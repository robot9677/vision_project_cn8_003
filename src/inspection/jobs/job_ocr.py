import pytesseract

def run_ocr(crop, cfg):
    try:
        txt = pytesseract.image_to_string(crop)
    except:
        txt = ""

    return True, {"ocr_text": txt.strip()}, "OK"


def eval_ocr(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    txt = metrics.get("ocr_text", "")

    if not txt:
        return False, "NG: OCR FAIL"

    prefix = cfg.get("expected_prefix", "")
    if prefix and not txt.startswith(prefix):
        return False, "NG: OCR INVALID"

    return True, f"OK: {txt}"