from functools import partial
from similarity import register
from .mapping_methods import neural_regression


# TODO: define rdm_compare but not computing rdm
# too complex to use 
# https://github.com/ColinConwell/DeepDive/blob/main/deepdive/feature_reduction.py


@register("postprocessing/mean_score")
def mean_score(scores):
    return scores.mean()


register = partial(
    register,
    postprocessing=["mean_score"]
)

score_types = ["pearson_r", "pearson_r2", "r2"]


for score_type in score_types:
    register(
        f"deepdive/neural_regression-alpha1-{score_type}-5folds_cv",
        partial(neural_regression, alphas=[1], score_type=score_type),
    )
    register(
        f"deepdive/neural_regression-alpha0-{score_type}-5folds_cv",
        partial(neural_regression, alphas=[0], score_type=score_type),
    )

    # TODO: cv_splits=None returns regression object...
    # register(
    #     "measure/deepdive/ridge-lambda1-pearson_r-no_cv",
    #     partial(neural_regression, cv_splits=None),
    # )