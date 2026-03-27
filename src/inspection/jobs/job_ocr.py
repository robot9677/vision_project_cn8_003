import pytesseract

def run_ocr(crop, cfg):
    try:
        txt = pytesseract.image_to_string(crop)
    except:
        txt = ""

    return True, {
        "ocr_text": txt.strip(),
        "_last_image": crop  
    }, "OK"


def eval_ocr(ok, metrics, reason, cfg, recipe_default, runtime_cfg):
    txt = str(metrics.get("ocr_text", "") or "").strip()

    if not txt:
        return False, "NG: OCR FAIL"

    # normalize (숫자만)
    txt_norm = "".join([c for c in txt if c.isdigit()])

    digits_only = bool(cfg.get("digits_only", False))
    exact_length = cfg.get("exact_length", None)

    if digits_only:
        if not txt_norm.isdigit() or not txt_norm:
            return False, "NG: OCR NOT DIGITS"

    if exact_length is not None:
        if len(txt_norm) != int(exact_length):
            return False, "NG: OCR LENGTH MISMATCH"

    # 저장
    metrics["ocr_text_norm"] = txt_norm

    return True, f"OK: {txt_norm}"