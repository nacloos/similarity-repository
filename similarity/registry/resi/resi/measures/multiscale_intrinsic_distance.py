from typing import Union

import numpy as np
import numpy.typing as npt
import torch
from resi.measures.utils import align_spatial_dimensions
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def imd_score(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    approximation_steps: int = 8000,
    n_repeat: int = 5,
) -> float:
    try:
        import msid
    except ImportError as e:
        print(
            "Install IMD from" "https://github.com/xgfs/imd.git." "Clone and cd into directory, then `pip install .`"
        )
        raise e

    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    # We use much higher defaults for the heat kernel approximation steps as the results
    # have very high variance otherwise. We also repeat the estimation to get an idea
    # about the variance of the score (TODO: report variance)
    scores = [msid.msid_score(R, Rp, niters=approximation_steps) for _ in range(n_repeat)]
    return float(np.mean(scores))


class IMDScore(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=imd_score,
            larger_is_more_similar=False,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,  # because default lambda=0
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
        approximation_steps: int = 8000,
        n_repeat: int = 5,
    ) -> float:
        if shape == "nchw":
            # Move spatial dimensions into the sample dimension
            # If not the same spatial dimension, resample via FFT.
            R, Rp = align_spatial_dimensions(R, Rp)
            shape = "nd"

        return self.sim_func(R, Rp, shape, approximation_steps, n_repeat)  # type:ignore
