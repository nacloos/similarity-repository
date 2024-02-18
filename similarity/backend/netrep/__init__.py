from functools import partial
import netrep.metrics

from similarity.backend.netrep import cka  # fix small bug in netrep.cka
from similarity import register

# TODO: specify that center_columns: bool = True, zero_pad: bool = True,
# or just do it again to make sure it's clear
# TODO: add reshape2d preprocessing
# TODO: specify interface when registering
# TODO: register github, paper, and author

register = partial(
    register,
    function=False,
    preprocessing=[
        "reshape2d",
        # "center_columns",
        # "zero_pad"
    ],
    interface={
        "__call__": "fit_score"
    }
)

score_methods = ["angular", "euclidean"]

for score_method in score_methods:
    register(
        f"measure.netrep.procrustes-{score_method}",
        partial(netrep.metrics.LinearMetric, alpha=1, score_method=score_method),
    )
    register(
        f"measure.netrep.cca-{score_method}",
        partial(netrep.metrics.LinearMetric, alpha=0, score_method=score_method)
    )
    register(
        f"measure.netrep.permutation-{score_method}",
        partial(netrep.metrics.PermutationMetric, score_method=score_method)
    )

register(
    "measure.netrep.cka-angular",
    cka.LinearCKA,
    interface={
        # TODO: raise error if try to call fit or fit_score (e.g. "Cross-validated score not available for CKA.")
        "__call__": "score"  # redirect __call__ to score method because fit_score is not implemented
    }
)
