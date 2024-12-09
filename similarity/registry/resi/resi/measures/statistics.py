from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import RSMSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def magnitude_difference(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    return abs(float(np.linalg.norm(R.mean(axis=0)) - np.linalg.norm(Rp.mean(axis=0))))


def magnitude_nrmse(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    di_bar = np.hstack(
        [
            np.linalg.norm(R, axis=1, ord=2, keepdims=True),
            np.linalg.norm(Rp, axis=1, ord=2, keepdims=True),
        ]
    ).mean(axis=1)
    rmse = np.sqrt(
        1 / 2 * ((np.linalg.norm(R, axis=1, ord=2) - di_bar) ** 2 + (np.linalg.norm(Rp, axis=1, ord=2) - di_bar) ** 2)
    )
    normalization = np.abs(np.linalg.norm(R, axis=1, ord=2) - np.linalg.norm(Rp, axis=1, ord=2))
    # this might create nans as normalization can theoretically be zero, but we fix this
    # by setting the nan values to zero (If there is no difference in the norm of the
    # instance in both representations, then the RMSE term will also be zero. We then
    # say that 0/0 = 0 variance.).
    per_instance_nrmse = rmse / normalization
    per_instance_nrmse[np.isnan(per_instance_nrmse)] = 0
    return float(per_instance_nrmse.mean())


def uniformity_difference(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)

    def uniformity(x, t=2):
        # TODO: sqeuclidean does not have a fast sklearn implementation. Is using euclidean and then squaring faster?
        pdist = sklearn.metrics.pairwise_distances(x, metric="sqeuclidean", n_jobs=n_jobs)
        return np.log(np.exp(-t * pdist).sum() / x.shape[0] ** 2)

    return float(abs(uniformity(R) - uniformity(Rp)))


def concentricity(x):
    return 1 - sklearn.metrics.pairwise_distances(x, x.mean(axis=0, keepdims=True), metric="cosine")


def concentricity_difference(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    return float(abs(concentricity(R).mean() - concentricity(Rp).mean()))


def concentricity_nrmse(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    alphai_bar = np.hstack(
        [
            concentricity(R),
            concentricity(Rp),
        ]
    ).mean(axis=1, keepdims=True)
    rmse = np.sqrt(((concentricity(R) - alphai_bar) ** 2 + (concentricity(Rp) - alphai_bar) ** 2) / 2)
    normalization = np.abs(concentricity(R) - concentricity(Rp))

    # this might create nans as normalization can theoretically be zero, but we fix this
    # by setting the nan values to zero (If there is no difference in the norm of the
    # instance in both representations, then the RMSE term will also be zero. We then
    # say that 0/0 = 0 variance.).
    per_instance_nrmse = rmse / normalization
    per_instance_nrmse[np.isnan(per_instance_nrmse)] = 0
    return float(per_instance_nrmse.mean())


class MagnitudeDifference(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=magnitude_difference,
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )


class UniformityDifference(RSMSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=uniformity_difference,
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=True,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.

            # RSM base measures should be invariant to this intheory, but Uniformity takes too long to reasonably handle it.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class ConcentricityDifference(RepresentationalSimilarityMeasure):
    def __init__(self):
        # different choice of inner/outer in __call__ should change these values...
        super().__init__(
            sim_func=concentricity_difference,
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=False,
        )
