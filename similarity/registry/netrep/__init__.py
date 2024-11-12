from functools import partial
import netrep.metrics

import similarity
from . import cka  # fix small bug in netrep.cka


similarity.register(
    "measure/netrep",
    {
        "paper_id": "williams2021", 
        "github": "https://github.com/ahwillia/netrep"
    }
)

register = partial(
    similarity.register,
    function=False,
    preprocessing=[
        "reshape2d",
        # don't center cols here, netrep is doing it in a cross-validated way
    ],
    interface={
        # allow the measure to be called like a function
        "__call__": "fit_score"
    }
)

score_methods = ["angular", "euclidean"]

for score_method in score_methods:
    register(
        f"measure/netrep/procrustes-{score_method}",
        partial(netrep.metrics.LinearMetric, alpha=1, score_method=score_method),
    )
    register(
        f"measure/netrep/cca-{score_method}",
        partial(netrep.metrics.LinearMetric, alpha=0, score_method=score_method)
    )
    register(
        f"measure/netrep/permutation-{score_method}",
        partial(netrep.metrics.PermutationMetric, score_method=score_method)
    )

register(
    "measure/netrep/cka-angular",
    cka.LinearCKA,
    interface={
        # raise error if try to call fit or fit_score (e.g. "Cross-validated score not available for CKA.")
        "__call__": "score"  # redirect __call__ to score method because fit_score is not implemented
    }
)


# TODO: register shape metric for other values of alpha?