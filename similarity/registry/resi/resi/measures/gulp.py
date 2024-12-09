import math
from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


# Copied from https://github.com/sgstepaniants/GULP/blob/main/distance_functions.py
def predictor_dist(A, B, evals_a=None, evecs_a=None, evals_b=None, evecs_b=None, lmbda=0):
    """
    Computes distance between best linear predictors on representations A and B
    """
    k, n = A.shape
    l, _ = B.shape
    assert k <= n
    assert l <= n

    if evals_a is None or evecs_a is None:
        evals_a, evecs_a = np.linalg.eigh(A @ A.T)
    if evals_b is None or evecs_b is None:
        evals_b, evecs_b = np.linalg.eigh(B @ B.T)

    evals_a = (evals_a + np.abs(evals_a)) / (2 * n)
    if lmbda > 0:
        inv_a_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_a])
    else:
        inv_a_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_a])

    evals_b = (evals_b + np.abs(evals_b)) / (2 * n)
    if lmbda > 0:
        inv_b_lmbda = np.array([1 / (x + lmbda) if x > 0 else 1 / lmbda for x in evals_b])
    else:
        inv_b_lmbda = np.array([1 / x if x > 0 else 0 for x in evals_b])

    T1 = np.sum(np.square(evals_a * inv_a_lmbda))
    T2 = np.sum(np.square(evals_b * inv_b_lmbda))

    cov_ab = A @ B.T / n
    T3 = np.trace(
        (np.diag(np.sqrt(inv_a_lmbda)) @ evecs_a.T)
        @ cov_ab
        @ (evecs_b @ np.diag(inv_b_lmbda) @ evecs_b.T)
        @ cov_ab.T
        @ (evecs_a @ np.diag(np.sqrt(inv_a_lmbda)))
    )

    return T1 + T2 - 2 * T3


# End of copy


def gulp(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    lmbda: float = 0,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    # The GULP paper assumes DxN matrices; we have NxD matrices.
    n = R.shape[0]
    rep1 = R.T
    rep2 = Rp.T
    # They further assume certain normalization (taken from https://github.com/sgstepaniants/GULP/blob/d572663911cf8724ed112ee566ca956089bfe678/cifar_experiments/compute_dists.py#L82C5-L89C54)
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep1 = math.sqrt(n) * rep1 / np.linalg.norm(rep1)
    # center and normalize
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    rep2 = math.sqrt(n) * rep2 / np.linalg.norm(rep2)

    return predictor_dist(rep1, rep2, lmbda=lmbda)  # type:ignore


class Gulp(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=gulp,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=True,  # because default lambda=0
            invariant_to_invertible_linear=True,
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
