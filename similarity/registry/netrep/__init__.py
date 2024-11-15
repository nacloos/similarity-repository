from functools import partial
import netrep
from netrep.metrics import LinearMetric, PermutationMetric, LinearCKA

import similarity
# from . import cka  # fix small bug in netrep.cka


# convert class interface into a function
def _linear_metric(X, Y, alpha, score_method):
    return LinearMetric(alpha=alpha, score_method=score_method).fit_score(X, Y)

def _permutation_metric(X, Y, score_method):
    return PermutationMetric(score_method=score_method).fit_score(X, Y)

def _cka(X, Y):
    return LinearCKA().score(X, Y)


score_methods = ["angular", "euclidean"]
for score_method in score_methods:
    similarity.register(
        f"netrep/LinearMetric_{score_method}",
        # f"measure/netrep/shape_metric-alpha={{alpha}}-distance={score_method}",
        partial(_linear_metric, score_method=score_method)
    )

    similarity.register(
        f"netrep/PermutationMetric_{score_method}",
        # f"measure/netrep/permutation_metric-distance={score_method}",
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


# similarity.register("measure/netrep/cka-kernel=linear-hsic=gretton-distance=angular", _cka)

# similarity.register("distance/netrep/cosine", netrep.utils.angular_distance)

# "kernel/netrep/rbf-sigma=1.0"
# "hsic/netrep/lange"
# "distance/netrep/cosine"
# "measure/netrep/cka-kernel=(rbf-sigma=1.0)-hsic=lange-distance=cosine"



# similarity.register(
#     "measure/netrep",
#     {
#         "paper_id": "williams2021", 
#         "github": "https://github.com/ahwillia/netrep"
#     }
# )

# register = partial(
#     similarity.register,
#     # function=False,
#     preprocessing=[
#         "reshape2d",
#         # don't center cols here, netrep is doing it in a cross-validated way
#     ],
#     interface={
#         # allow the measure to be called like a function
#         "__call__": "fit_score"
#     }
# )
# score_methods = ["angular", "euclidean"]

# for score_method in score_methods:
#     register(
#         f"measure/netrep/procrustes-{score_method}",
#         partial(netrep.metrics.LinearMetric, alpha=1, score_method=score_method),
#     )
#     register(
#         f"measure/netrep/cca-{score_method}",
#         partial(netrep.metrics.LinearMetric, alpha=0, score_method=score_method)
#     )
#     register(
#         f"measure/netrep/permutation-{score_method}",
#         partial(netrep.metrics.PermutationMetric, score_method=score_method)
#     )

# register(
#     "measure/netrep/cka-angular",
#     cka.LinearCKA,
#     interface={
#         # raise error if try to call fit or fit_score (e.g. "Cross-validated score not available for CKA.")
#         "__call__": "score"  # redirect __call__ to score method because fit_score is not implemented
#     }
# )


# TODO: register shape metric for other values of alpha?