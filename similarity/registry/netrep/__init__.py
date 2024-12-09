# https://github.com/ahwillia/netrep
from functools import partial
import netrep
from netrep.metrics import LinearMetric, PermutationMetric, LinearCKA

import similarity


# convert class interface into a function
def _linear_metric(X, Y, alpha=1.0, score_method="angular"):
    return LinearMetric(alpha=alpha, score_method=score_method).fit_score(X, Y)

def _permutation_metric(X, Y, score_method="angular"):
    return PermutationMetric(score_method=score_method).fit_score(X, Y)

def _cka(X, Y):
    return LinearCKA().score(X, Y)


score_methods = ["angular", "euclidean"]
for score_method in score_methods:
    similarity.register(
        f"netrep/LinearMetric_{score_method}",
        partial(_linear_metric, score_method=score_method)
    )

    similarity.register(
        f"netrep/PermutationMetric_{score_method}",
        partial(_permutation_metric, score_method=score_method)
    )

similarity.register("netrep/LinearCKA", _cka)

# register the linear kernel code copied from LinearCKA
similarity.register(
    "kernel/netrep/linear",
    # same as line 46 of netrep/metrics/cka.py
    lambda X: X @ X.T
)

similarity.register("distance/netrep/angular", netrep.utils.angular_distance)
