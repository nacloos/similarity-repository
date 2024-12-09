from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def eigenspace_overlap_score(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    u, _, _ = np.linalg.svd(R)
    v, _, _ = np.linalg.svd(Rp)
    u = u[:, : np.linalg.matrix_rank(R)]
    v = v[:, : np.linalg.matrix_rank(Rp)]
    return 1 / np.max([R.shape[1], Rp.shape[1]]) * (np.linalg.norm(u.T @ v, ord="fro") ** 2)


class EigenspaceOverlapScore(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=eigenspace_overlap_score,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=True,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=False,
        )

    def __call__(self, R: torch.Tensor | npt.NDArray, Rp: torch.Tensor | npt.NDArray, shape: SHAPE_TYPE) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape)
