from collections import deque
import numpy as np

class TemporalMeanFilter:
    def __init__(self, win=5):
        self.win = int(win)
        self.buf = deque(maxlen=self.win)

    def reset(self):
        self.buf.clear()

    def update(self, value: float) -> float:
        self.buf.append(float(value))
        return float(np.mean(self.buf)) if self.buf else float(value)
