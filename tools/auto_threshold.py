import os
import cv2
import numpy as np

BASE = "data/dataset"

OK_DIR = os.path.join(BASE, "OK")
NG_DIR = os.path.join(BASE, "NG")

def load_scores(folder):

    scores = []

    for f in os.listdir(folder):
        if not f.endswith("_crop.png"):
            continue

        p = os.path.join(folder, f)
        img = cv2.imread(p, cv2.IMREAD_GRAYSCALE)

        if img is None:
            continue

        scores.append(float(img.mean()))

    return scores


ok_scores = load_scores(OK_DIR)
ng_scores = load_scores(NG_DIR)

print("OK samples :", len(ok_scores))
print("NG samples :", len(ng_scores))

ok_mean = np.mean(ok_scores)
ng_mean = np.mean(ng_scores)

print("OK mean :", ok_mean)
print("NG mean :", ng_mean)

thr = (ok_mean + ng_mean) / 2

print("")
print("recommended threshold :", thr)