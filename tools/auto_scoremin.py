import os, cv2, numpy as np

ROI_ID = 1
TEMPLATE = f"data/templates/tape_ok_ROI{ROI_ID}.png"
OK_DIR = "data/dataset/OK"
NG_DIR = "data/dataset/NG"

def list_crops(d):
    return [os.path.join(d,f) for f in os.listdir(d) if f.endswith("_crop.png")]

tmpl0 = cv2.imread(TEMPLATE, cv2.IMREAD_GRAYSCALE)
if tmpl0 is None:
    raise SystemExit(f"template load fail: {TEMPLATE}")

def best_score(img, s_min=0.85, s_max=1.15, s_step=0.05):
    best = -1.0
    s = s_min
    while s <= s_max + 1e-9:
        tw = int(tmpl0.shape[1]*s); th = int(tmpl0.shape[0]*s)
        if tw < 5 or th < 5:
            s += s_step; continue
        tmpl = cv2.resize(tmpl0, (tw, th), interpolation=cv2.INTER_AREA)
        if img.shape[0] < tmpl.shape[0] or img.shape[1] < tmpl.shape[1]:
            s += s_step; continue
        res = cv2.matchTemplate(img, tmpl, cv2.TM_CCOEFF_NORMED)
        _, sc, _, _ = cv2.minMaxLoc(res)
        if sc > best: best = float(sc)
        s += s_step
    return best

def scores(paths):
    out=[]
    for p in paths:
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)
        if img is None: 
            continue
        out.append(best_score(img))
    return out

ok = scores(list_crops(OK_DIR))
ng = scores(list_crops(NG_DIR))

print("OK n:", len(ok), "min/mean/max:", min(ok), np.mean(ok), max(ok))
print("NG n:", len(ng), "min/mean/max:", min(ng), np.mean(ng), max(ng))

# 추천: OK 하위 5%와 NG 상위 5% 사이 중간값
ok_p5 = float(np.percentile(ok, 5))
ng_p95 = float(np.percentile(ng, 95))
rec = (ok_p5 + ng_p95) / 2.0

print("ok_p5:", ok_p5)
print("ng_p95:", ng_p95)
print("recommended score_min:", rec)