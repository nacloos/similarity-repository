import numpy as np

from brainscore.metrics.regression import CrossRegressedCorrelation, pls_regression, pearsonr_correlation
from brainscore.metrics.rdm import RDMCrossValidated
from brainio.assemblies import NeuroidAssembly


def numpy_to_brainio(X, Y):
    """
    BrainScore measure takes NeuroidAssembly objects
    """
    def _convert(X):
        assert len(X.shape) == 2, X.shape
        return NeuroidAssembly(
            X,
            # filled with dummy values
            coords={
                'stimulus_id': ('presentation', np.arange(X.shape[0])),
                'object_name': ('presentation', ['']*X.shape[0]),
                'neuroid_id': ('neuroid', np.arange(X.shape[1])),
                'region': ('neuroid', [0] * X.shape[1])
            },
            dims=['presentation', 'neuroid']
        )
    return _convert(X), _convert(Y)


def aggregate_score(score):
    if "aggregation" in score.coords:
        return score.sel(aggregation='center').values.item()
    else:
        return score.values.item()
