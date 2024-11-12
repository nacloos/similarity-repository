from functools import partial
from similarity import register
from .mapping_methods import neural_regression


register(
    "measure/deepdive",
    {
        "paper_id": "conwell2023",
        "github": "https://github.com/ColinConwell/DeepDive"
    }
)


@register("postprocessing/mean_score")
def mean_score(scores):
    return scores.mean()


register = partial(
    register,
    function=True,
    preprocessing=["reshape2d"],
    postprocessing=["mean_score"]
)

score_types = ["pearson_r", "pearson_r2", "r2"]


for score_type in score_types:
    register(
        f"measure/deepdive/ridge-lambda1-{score_type}#5folds_cv",
        partial(neural_regression, alphas=[1], score_type=score_type),
    )
    register(
        f"measure/deepdive/linreg-{score_type}#5folds_cv",
        partial(neural_regression, alphas=[0], score_type=score_type),
    )

    # TODO: cv_splits=None returns regression object...
    # register(
    #     "measure/deepdive/ridge-lambda1-pearson_r#no_cv",
    #     partial(neural_regression, cv_splits=None),
    # )