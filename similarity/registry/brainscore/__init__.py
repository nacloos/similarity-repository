# Cloned the brainscore repository and modified the setup.py to allow newer versions of scikit-learn (https://github.com/brain-score/brain-score/issues/327)
# Deleted brainscore tests because don't want them to be run by pytest
import sys
from pathlib import Path
# make brainscore importable by adding current folder to path
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from functools import partial
import numpy as np

import similarity

from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, linear_regression, ridge_regression, pearsonr_correlation
from brainscore.metrics.rdm import RDMCrossValidated, RDMMetric
from brainscore.metrics.correlation import Correlation
from brainscore.metrics.cka import CKAMetric
from brainio.assemblies import NeuroidAssembly


similarity.register(
    "measure/brainscore",
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


# TODO: register functions
register = partial(
    similarity.register,
    preprocessing=[numpy_to_brainio],
    postprocessing=[aggregate_score],
)


register("brainscore/cka", lambda x, y: CKAMetric()(x, y))
register("brainscore/rsa-correlation-spearman", lambda x, y: RDMMetric()(x, y))
register("brainscore/correlation", lambda x, y: Correlation()(x, y))


regression_methods = {
    "linear_regression": linear_regression,
    "ridge_regression": ridge_regression,
    "pls_regression": pls_regression
}
for k, regression in regression_methods.items():
    # TODO: specify stratification coord (e.g. 'object_name')?
    register(
        f"brainscore/{k}-pearsonr_correlation",
        lambda x, y: CrossRegressedCorrelation(
            regression=regression(),
            correlation=pearsonr_correlation()
        )(x, y)
    )

    # TODO: set random seed
    register(
        f"brainscore/{k}-pearsonr_correlation-5folds_cv",
        lambda x, y: CrossRegressedCorrelation(
            regression=regression(),
            correlation=pearsonr_correlation(),
            crossvalidation_kwargs={
                "kfold": True,
                "splits": 5,
                "stratification_coord": False
            }
        )(x, y)
    )

