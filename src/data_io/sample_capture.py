import os
import json
import time
import cv2


def prune_snapshots(path, max_keep=200):
    try:
        files = []
        for fn in os.listdir(path):
            if fn.endswith(".png") or fn.endswith(".jpg"):
                p = os.path.join(path, fn)
                files.append((os.path.getmtime(p), p))
        files.sort(reverse=True)
        for _, p in files[max_keep:]:
            try:
                os.remove(p)
            except Exception:
                pass
    except Exception:
        pass


def prune_manifests(dir_path, keep=200):
    try:
        items = []
        for fn in os.listdir(dir_path):
            if fn.endswith(".json"):
                p = os.path.join(dir_path, fn)
                items.append((os.path.getmtime(p), p))
        items.sort(reverse=True)

        for _, jpath in items[keep:]:
            try:
                with open(jpath, "r", encoding="utf-8") as f:
                    meta = json.load(f)
                for k in ("raw", "overlay", "crop"):
                    p = meta.get(k)
                    if p and os.path.exists(p):
                        try:
                            os.remove(p)
                        except Exception:
                            pass
                os.remove(jpath)
            except Exception:
                try:
                    os.remove(jpath)
                except Exception:
                    pass
    except Exception:
        pass


def _get_roi_by_id(roi_mgr, roi_id: int):
    try:
        for r in getattr(roi_mgr, "rois", []):
            if int(r.get("id", -1)) == int(roi_id):
                return r
    except Exception:
        pass
    return None


def crop_roi(gray8, roi_mgr, roi_id: int):
    r = _get_roi_by_id(roi_mgr, roi_id)
    if r is None or gray8 is None:
        return None

    x = int(r.get("x", 0))
    y = int(r.get("y", 0))
    w = int(r.get("w", 0))
    h = int(r.get("h", 0))

    if w <= 0 or h <= 0:
        return None

    H, W = gray8.shape[:2]
    x1 = max(0, min(x, W - 1))
    y1 = max(0, min(y, H - 1))
    x2 = max(x1 + 1, min(W, x1 + w))
    y2 = max(y1 + 1, min(H, y1 + h))

    crop = gray8[y1:y2, x1:x2]
    if crop is None or crop.size == 0:
        return None
    return crop


def handle_sample_keys(
    key,
    frame_gray8,
    vis_bgr,
    *,
    edit_mode,
    roi_mgr,
    data_dir,
    snapshot_keep=10,
):
    if edit_mode:
        return False, None

    if key not in (ord("t"), ord("T"), ord("k"), ord("K"), ord("n"), ord("N")):
        return False, None

    try:
        sel = roi_mgr.get_selected()
    except Exception:
        sel = None

    roi_id = int(sel["id"]) if (isinstance(sel, dict) and sel.get("id") is not None) else 1

    is_t = key in (ord("t"), ord("T"))
    is_k = key in (ord("k"), ord("K"))
    is_n = key in (ord("n"), ord("N"))

    if is_t:
        crop = crop_roi(frame_gray8, roi_mgr, roi_id)
        if crop is not None:
            tpath = os.path.join(data_dir, "templates", f"tape_ok_ROI{roi_id}.png")
            cv2.imwrite(tpath, crop)
            return True, f"[TEMPLATE SAVED] {tpath}"
        return True, "[TEMPLATE SAVE FAILED] crop is empty"

    tag = "OK" if is_k else "NG"
    out_dir = os.path.join(data_dir, "dataset", tag)
    os.makedirs(out_dir, exist_ok=True)

    ts = time.strftime("%Y%m%d_%H%M%S")
    stem = f"cap_{ts}_ROI{roi_id}"

    raw_path = os.path.join(out_dir, f"{stem}_raw.png")
    ov_path = os.path.join(out_dir, f"{stem}_overlay.png")
    crop_path = os.path.join(out_dir, f"{stem}_crop.png")
    json_path = os.path.join(out_dir, f"{stem}.json")

    cv2.imwrite(raw_path, frame_gray8)
    cv2.imwrite(ov_path, vis_bgr)

    crop = crop_roi(frame_gray8, roi_mgr, roi_id)
    if crop is not None:
        cv2.imwrite(crop_path, crop)

    meta = {
        "ts": ts,
        "roi_id": roi_id,
        "tag": tag,
        "raw": raw_path,
        "overlay": ov_path,
        "crop": crop_path if crop is not None else None,
    }

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(meta, f, ensure_ascii=False, indent=2)

    prune_snapshots(out_dir, max_keep=snapshot_keep)
    prune_manifests(out_dir, keep=snapshot_keep)

    return True, f"[SAMPLE SAVED] {tag} ROI{roi_id}"