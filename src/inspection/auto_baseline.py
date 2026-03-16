import json
import numpy as np
from pathlib import Path


class AutoBaseline:

    def __init__(self, save_path):
        self.save_path = Path(save_path)
        self.samples = {}

    def add_sample(self, roi_name, feature_name, value):

        key = f"{roi_name}:{feature_name}"

        if key not in self.samples:
            self.samples[key] = []

        self.samples[key].append(float(value))

    def compute(self):

        result = {}

        for key, values in self.samples.items():

            arr = np.array(values)

            mean = float(np.mean(arr))
            std = float(np.std(arr))

            min_v = float(np.min(arr))
            max_v = float(np.max(arr))

            low = mean - 2 * std
            high = mean + 2 * std

            roi, feature = key.split(":")

            if roi not in result:
                result[roi] = {}

            result[roi][feature] = {
                "mean": mean,
                "std": std,
                "min": min_v,
                "max": max_v,
                "low": low,
                "high": high
            }

        return result

    def save(self):

        data = self.compute()

        self.save_path.parent.mkdir(parents=True, exist_ok=True)

        with open(self.save_path, "w") as f:
            json.dump(data, f, indent=2)

        print(f"[AUTO BASELINE SAVED] {self.save_path}")
        print(json.dumps(data, indent=2))