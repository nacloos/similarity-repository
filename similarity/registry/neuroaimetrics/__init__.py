# https://github.com/anshksoni/NeuroAIMetrics
from functools import partial
from dataclasses import dataclass
import numpy as np

from .metrics import all_metrics

import similarity


# def make_measure(name):
#     def _measure(X: np.ndarray, Y: np.ndarray, **kwargs) -> float:
#         score = all_metrics[name](X, Y, **kwargs)
#         # score is a list
#         score = np.mean(score)
#         return float(score)

#     return _measure


def measure(X, Y, name, **kwargs):
    score = all_metrics[name](X, Y, **kwargs)
    # score is a list
    return np.mean(score)


for name in all_metrics.keys():
    similarity.register(
        f"neuroaimetrics/{name}",
        partial(measure, name=name)
    )
