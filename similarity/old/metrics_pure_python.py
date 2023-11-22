from functools import partial

import netrep.metrics
import rsatoolbox

import similarity.cka
import similarity.rsa

from similarity.metric import Metric
from similarity.processing import angular_dist_to_score, flatten_3d_to_2d, angular_dist
from config_utils.dict_module import DictModule, DictSequential


def make_metric(metric, preprocessing=None, postprocessing=None,
                call_key="fit_score") -> Metric:
    preprocessing = [] if preprocessing is None else preprocessing
    postprocessing = [] if postprocessing is None else postprocessing

    if call_key is None:
        _fit_score = partial(metric)
    elif call_key == "fit_score":
        _fit_score = partial(metric.fit_score)
    else:
        raise ValueError(f"Invalid call_key: {call_key}")

    def fit_score(X, Y):
        for p in preprocessing:
            X, Y = p(X, Y)

        # TODO: what if a metric outputs additional inputs or if name of inputs is differnt
        # => useful to use DictModule
        score = _fit_score(X=X, Y=Y)

        for p in postprocessing:
            score = p(score)

        return score

    # fit_score = DictSequential(
    #     DictModule(
    #         _fit_score,
    #         in_keys=["X", "Y"],
    #         out_keys=["score"]
    #     )
    # )

    return Metric(
        metric=metric,
        fit_score=fit_score
    )


Procrustes = partial(
    make_metric,
    metric=netrep.metrics.LinearMetric(alpha=1),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist_to_score]
)

CCA = partial(
    make_metric,
    metric=netrep.metrics.LinearMetric(alpha=0),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist_to_score]
)

CKA = partial(
    make_metric,
    metric=partial(similarity.cka.linear_CKA),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist, angular_dist_to_score],
    call_key=None
)

RSA = partial(
    make_metric,
    metric=partial(
        similarity.rsa.compute_rsa,
        rdm_method='euclidean',
        compare_method='cosine'
    ),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist, angular_dist_to_score],
    call_key=None
)

# can't extract a value from previous config
# RSA2 = partial(make_metric, metric=RSA.metric, ...)

# can't use base config
# BaseCKA = partial(make_metric, metric=partial(similarity.cka.linear_CKA))
# RSA1 = partial(BaseCKA, metric.rdm_method="")
# RS12 = ...
# have to write a for loop
# for rdm_method in ["euclidean", "correlation"]:
#     CKA = partial(
#         make_metric, 
#         metric=partial(similarity.cka.linear_CKA, rdm_method=rdm_method
#     )


if __name__ == "__main__":
    import numpy as np

    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)

    metric = Procrustes()
    print(metric.fit_score(X=X, Y=Y))

    metric = CCA()
    print(metric.fit_score(X=X, Y=Y))

    metric = CKA()
    print(metric.fit_score(X=X, Y=Y))

    metric = RSA()
    print(metric.fit_score(X=X, Y=Y))