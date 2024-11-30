# https://github.com/anshksoni/NeuroAIMetrics
from functools import partial
from dataclasses import dataclass
import numpy as np

from .metrics import all_metrics

import similarity


def make_measure(name):
    def _measure(X: np.ndarray, Y: np.ndarray) -> float:
        score = all_metrics[name](X, Y)
        # score is a list
        score = np.mean(score)
        return float(score)

    return _measure


for name in all_metrics.keys():
    similarity.register(
        f"neuroaimetrics/{name}",
        make_measure(name),
        function=True
    )
