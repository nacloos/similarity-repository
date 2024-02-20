from functools import partial
import numpy as np

import similarity

from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, linear_regression, ridge_regression, pearsonr_correlation
from brainscore.metrics.rdm import RDMCrossValidated, RDMMetric
from brainscore.metrics.correlation import Correlation
from brainio.assemblies import NeuroidAssembly


similarity.register(
    "measure.brainscore",
    {
        "paper_id": "schrimpf2018",
        "github": "https://github.com/brain-score/vision"
    }
)


def numpy_to_brainio(X):
    """
    Helper function to convert numpy arrays to NeuroidAssembly objects, which are required by BrainScore measures
    """
    if isinstance(X, NeuroidAssembly):
        return X

    assert len(X.shape) == 2, X.shape
    return NeuroidAssembly(
        X,
        # fill with dummy values
        coords={
            'stimulus_id': ('presentation', np.arange(X.shape[0])),
            'object_name': ('presentation', ['']*X.shape[0]),
            'neuroid_id': ('neuroid', np.arange(X.shape[1])),
            'region': ('neuroid', [0] * X.shape[1])
        },
        dims=['presentation', 'neuroid']
    )


def aggregate_score(score):
    """
    Helper function to aggregate BrainScore scores into a single scalar value
    """
    if "aggregation" in score.coords:
        return score.sel(aggregation='center').values.item()
    else:
        return score.values.item()


register = partial(
    similarity.register,
    function=False,
    preprocessing=[
        "reshape2d",
        numpy_to_brainio
    ],
    postprocessing=[
        aggregate_score
    ],
    interface={
        "fit_score": "__call__"
    }
)


regression_methods = {
    "linreg": linear_regression,
    "ridge-lambda1": ridge_regression,
    "pls": pls_regression
}
for k, regression in regression_methods.items():
    # TODO: specify stratification coord (e.g. 'object_name')?
    register(
        f"measure.brainscore.{k}-pearson_r#10splits_90/10ratio_cv",
        partial(
            CrossRegressedCorrelation,
            regression=regression(),
            correlation=pearsonr_correlation()
        )
    )

    # TODO: set random seed
    register(
        f"measure.brainscore.{k}-pearson_r#5folds_cv",
        partial(
            CrossRegressedCorrelation,
            regression=regression(),
            correlation=pearsonr_correlation(),
            crossvalidation_kwargs={
                "kfold": True,
                "splits": 5,
                "stratification_coord": False
            }
        )
    )

register(
    "measure.brainscore.rsa-correlation-spearman",
    RDMMetric
)

register(
    "measure.brainscore.correlation",
    Correlation
)
