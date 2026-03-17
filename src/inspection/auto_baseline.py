import json
from pathlib import Path


class AutoBaseline:
    """
    현장형 baseline 관리자
    - raw sample 배열 저장 안 함
    - ROI/feature별 count, mean, m2(분산 누적치)만 저장
    - save_path JSON 1개만 overwrite
    """

    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.stats = {}

    def _key(self, roi_name, feature_name):
        return f"{roi_name}:{feature_name}"

    def reset(self):
        self.stats = {}

    def add_sample(self, roi_name, feature_name, value):
        key = self._key(roi_name, feature_name)
        x = float(value)

        if key not in self.stats:
            self.stats[key] = {
                "count": 0,
                "mean": 0.0,
                "m2": 0.0,
                "min": x,
                "max": x,
            }

        s = self.stats[key]
        s["count"] += 1

        delta = x - s["mean"]
        s["mean"] += delta / s["count"]
        delta2 = x - s["mean"]
        s["m2"] += delta * delta2

        if x < s["min"]:
            s["min"] = x
        if x > s["max"]:
            s["max"] = x

    def compute(self):
        result = {}

        for key, s in self.stats.items():
            roi, feature = key.split(":")

            count = int(s["count"])
            mean = float(s["mean"])
            min_v = float(s["min"])
            max_v = float(s["max"])

            if count > 1:
                var = float(s["m2"]) / float(count - 1)
            else:
                var = 0.0

            std = var ** 0.5

            low = mean - 2.0 * std
            high = mean + 2.0 * std

            if roi not in result:
                result[roi] = {}

            result[roi][feature] = {
                "count": count,
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "low": low,
                "high": high,
            }

        return result

    def save(self):
        data = self.compute()
        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=2)

        print(f"[BASELINE SAVED] {self.save_path}")

    def load(self):
        if not self.save_path.exists():
            return False

        with open(self.save_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        self.stats = {}

        for roi_name, feat_map in data.items():
            for feature_name, v in feat_map.items():
                count = int(v.get("count", 0))
                mean = float(v.get("mean", 0.0))
                std = float(v.get("std", 0.0))
                min_v = float(v.get("min", mean))
                max_v = float(v.get("max", mean))

                if count <= 1:
                    m2 = 0.0
                else:
                    m2 = (std ** 2) * (count - 1)

                key = self._key(roi_name, feature_name)
                self.stats[key] = {
                    "count": count,
                    "mean": mean,
                    "m2": m2,
                    "min": min_v,
                    "max": max_v,
                }

        return True

    def update_from_ok_result(self, roi_name, feature_name, value, max_count=200):
        """
        기존 baseline에 새 OK 결과 1개 반영
        max_count를 넘으면 오래된 이력 효과를 약하게 만들어 count를 제한
        """
        key = self._key(roi_name, feature_name)
        x = float(value)

        if key not in self.stats:
            self.add_sample(roi_name, feature_name, x)
            return

        s = self.stats[key]

        if s["count"] >= max_count:
            # count를 줄여서 오래된 이력 영향 완만하게 감소
            s["count"] = max(20, int(max_count * 0.7))
            s["m2"] *= 0.7

        self.add_sample(roi_name, feature_name, x)