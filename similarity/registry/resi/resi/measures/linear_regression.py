from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import center_columns
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def linear_reg(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    R = center_columns(R)
    Rp = center_columns(Rp)
    Rp_orthonormal_base = Rp @ scipy.linalg.inv(  # type:ignore
        scipy.linalg.sqrtm(Rp.T @ Rp)  # type:ignore
    )
    return float((np.linalg.norm(Rp_orthonormal_base.T @ R, ord="fro") ** 2) / (np.linalg.norm(R, ord="fro") ** 2))


class LinearRegression(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=linear_reg,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=False,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
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
