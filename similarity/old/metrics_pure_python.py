from functools import partial

import netrep.measures
import rsatoolbox

import similarity.cka
import similarity.rsa

from similarity.measure import Measure
from similarity.processing import angular_dist_to_score, flatten_3d_to_2d, angular_dist
from config_utils.dict_module import DictModule, DictSequential


def make_measure(measure, preprocessing=None, postprocessing=None,
                call_key="fit_score") -> Measure:
    preprocessing = [] if preprocessing is None else preprocessing
    postprocessing = [] if postprocessing is None else postprocessing

    if call_key is None:
        _fit_score = partial(measure)
    elif call_key == "fit_score":
        _fit_score = partial(measure.fit_score)
    else:
        raise ValueError(f"Invalid call_key: {call_key}")

    def fit_score(X, Y):
        for p in preprocessing:
            X, Y = p(X, Y)

        # TODO: what if a measure outputs additional inputs or if name of inputs is differnt
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

    return Measure(
        measure=measure,
        fit_score=fit_score
    )


Procrustes = partial(
    make_measure,
    measure=netrep.measures.Linearmeasure(alpha=1),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist_to_score]
)

CCA = partial(
    make_measure,
    measure=netrep.measures.Linearmeasure(alpha=0),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist_to_score]
)

CKA = partial(
    make_measure,
    measure=partial(similarity.cka.linear_CKA),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist, angular_dist_to_score],
    call_key=None
)

RSA = partial(
    make_measure,
    measure=partial(
        similarity.rsa.compute_rsa,
        rdm_method='euclidean',
        compare_method='cosine'
    ),
    preprocessing=[flatten_3d_to_2d],
    postprocessing=[angular_dist, angular_dist_to_score],
    call_key=None
)

# can't extract a value from previous config
# RSA2 = partial(make_measure, measure=RSA.measure, ...)

# can't use base config
# BaseCKA = partial(make_measure, measure=partial(similarity.cka.linear_CKA))
# RSA1 = partial(BaseCKA, measure.rdm_method="")
# RS12 = ...
# have to write a for loop
# for rdm_method in ["euclidean", "correlation"]:
#     CKA = partial(
#         make_measure, 
#         measure=partial(similarity.cka.linear_CKA, rdm_method=rdm_method
#     )


if __name__ == "__main__":
    import numpy as np

    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)

    measure = Procrustes()
    print(measure.fit_score(X=X, Y=Y))

    measure = CCA()
    print(measure.fit_score(X=X, Y=Y))

    measure = CKA()
    print(measure.fit_score(X=X, Y=Y))

    measure = RSA()
    print(measure.fit_score(X=X, Y=Y))