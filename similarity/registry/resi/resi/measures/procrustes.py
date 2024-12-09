import warnings
from typing import Optional
from typing import Tuple
from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize
import torch
from resi.measures.utils import adjust_dimensionality
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import center_columns
from resi.measures.utils import flatten
from resi.measures.utils import normalize_matrix_norm
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def orthogonal_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)[1]
    squared_dist = -2 * nucnorm + np.linalg.norm(R, ord="fro") ** 2 + np.linalg.norm(Rp, ord="fro") ** 2
    if squared_dist < 0:
        warnings.warn(
            f"Squared Orthogonal Procrustes distance is less than 0, but small, likely due to numerical errors. "
            f"Exact value={squared_dist}. Rounding to zero."
        )
        squared_dist = 0
    return np.sqrt(squared_dist)


def procrustes_size_and_shape_distance(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Same setup as Williams et al., 2021 for the rotation invariant metric"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    return orthogonal_procrustes(R, Rp, "nd")


def orthogonal_procrustes_centered_and_normalized(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Same setup as Ding et al., 2021"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)
    return orthogonal_procrustes(R, Rp, "nd")


def permutation_procrustes(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    optimal_permutation_alignment: Optional[Tuple[npt.NDArray, npt.NDArray]] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    if not optimal_permutation_alignment:
        PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments
        optimal_permutation_alignment = (PR, PRp)
    PR, PRp = optimal_permutation_alignment
    return float(np.linalg.norm(R[:, PR] - Rp[:, PRp], ord="fro"))


def permutation_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments

    aligned_R = R[:, PR]
    aligned_Rp = Rp[:, PRp]

    # matrices are already normalized so no division necessary
    corr = np.trace(aligned_R.T @ aligned_Rp)

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return np.arccos(np.clip(corr, -1.0, 1.0))


def orthogonal_angular_shape_metric(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    R, Rp = normalize_matrix_norm(R), normalize_matrix_norm(Rp)

    Qstar, nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)
    # matrices are already normalized so no division necessary
    corr = np.trace(Qstar.T @ R.T @ Rp)  # = \langle RQ, R' \rangle

    # From https://github.com/ahwillia/netrep/blob/0f3d825aad58c6d998b44eb0d490c0c5c6251fc9/netrep/utils.py#L107  # noqa: E501
    # numerical precision issues require us to clip inputs to arccos
    return float(np.arccos(np.clip(corr, -1.0, 1.0)))


def orthogonal_angular_shape_metric_centered(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Williams et al., 2021 version"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = center_columns(R), center_columns(Rp)
    return orthogonal_angular_shape_metric(R, Rp, "nd")


def aligned_cossim(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    align, _ = scipy.linalg.orthogonal_procrustes(R, Rp)

    R_aligned = R @ align
    sum_cossim = 0
    for r, rp in zip(R_aligned, Rp):
        sum_cossim += r.dot(rp) / (np.linalg.norm(r) * np.linalg.norm(rp))
    return sum_cossim / R.shape[0]


def permutation_aligned_cossim(R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)

    PR, PRp = scipy.optimize.linear_sum_assignment(R.T @ Rp, maximize=True)  # returns column assignments
    R_aligned = R[:, PR]
    Rp_aligned = Rp[:, PRp]

    sum_cossim = 0
    for r, rp in zip(R_aligned, Rp_aligned):
        sum_cossim += r.dot(rp) / (np.linalg.norm(r) * np.linalg.norm(rp))
    return sum_cossim / R.shape[0]


class ProcrustesSizeAndShapeDistance(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=procrustes_size_and_shape_distance,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
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
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class OrthogonalProcrustesCenteredAndNormalized(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=orthogonal_procrustes_centered_and_normalized,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
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


class PermutationProcrustes(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=permutation_procrustes,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=False,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)


class OrthogonalAngularShapeMetricCentered(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=orthogonal_angular_shape_metric_centered,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
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


class AlignedCosineSimilarity(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=aligned_cossim,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)
