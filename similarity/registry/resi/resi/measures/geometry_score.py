from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def geometry_score(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray], shape: SHAPE_TYPE, **kwargs
) -> float:
    try:
        import gs
    except ImportError as e:
        print(
            "Install the geometry score from"
            "https://github.com/KhrulkovV/geometry-score."
            "Clone and cd into directory, then `pip install .`"
        )
        raise e

    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    rlt_R = gs.rlts(R, **kwargs)
    mrlt_R = np.mean(rlt_R, axis=0)

    rlt_Rp = gs.rlts(Rp, **kwargs)
    mrlt_Rp = np.mean(rlt_Rp, axis=0)

    return float(np.sum((mrlt_R - mrlt_Rp) ** 2))


class GeometryScore(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=geometry_score,
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
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
