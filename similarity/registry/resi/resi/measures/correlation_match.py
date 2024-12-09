from typing import Optional
from typing import Union

import numpy.typing as npt
import scipy.optimize
import scipy.spatial.distance
import sklearn.metrics
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import center_columns
from resi.measures.utils import flatten
from resi.measures.utils import RSMSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def hard_correlation_match(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)

    # n_jobs only works if metric is in PAIRWISE_DISTANCES as defined in sklearn, i.e., not for correlation.
    # But correlation = 1 - cosine dist of row-centered data. Since we are using R^T, Rp^T, the columns of R, Rp must
    # be centered. This is already true from the above preprocessing.
    corr_matrix = 1 - sklearn.metrics.pairwise_distances(R.T, Rp.T, metric="cosine", n_jobs=n_jobs)

    # Let D = R.shape[1], Dp = Rp.shape[1] (the number of neurons).
    # Wlog, let D < Dp. Then all neurons of R get matched to a neuron of Rp, but
    # not all neurons of Rp are considered for the final score. The aligned matrices
    # will both have D neurons, the rest is discarded.
    # PR indexes the best-matched neurons of R, PRp are the corresponding matches in
    # Rp.
    PR, PRp = scipy.optimize.linear_sum_assignment(corr_matrix, maximize=True)
    return corr_matrix[PR, PRp].mean()


def soft_correlation_match(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)

    # n_jobs only works if metric is in PAIRWISE_DISTANCES as defined in sklearn, i.e., not for correlation.
    # But correlation = 1 - cosine dist of row-centered data. Since we are using R^T, Rp^T, the columns of R, Rp must
    # be centered. This is already true from the above preprocessing.
    corr_matrix = 1 - sklearn.metrics.pairwise_distances(R.T, Rp.T, metric="cosine", n_jobs=n_jobs)

    # Different to "hard" mode, all neurons of the first representation R are
    # considered, i.e., they will have a match with some neuron of Rp, but not
    # necessarily do all Rp neurons have a match. This means that this raw score is
    # asymmetric. We report the average between both directions to symmetrize it.
    score1 = corr_matrix.max(axis=1).mean()
    score2 = corr_matrix.max(axis=0).mean()
    return (score1 + score2) / 2


class HardCorrelationMatch(RSMSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=hard_correlation_match,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class SoftCorrelationMatch(RSMSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=soft_correlation_match,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)
