from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
from loguru import logger
from resi.measures.utils import double_center
from resi.measures.utils import flatten
from resi.measures.utils import RSMSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def distance_correlation(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)

    logger.info(f"Starting RSMs with {n_jobs=}")
    S = sklearn.metrics.pairwise_distances(R, metric="euclidean", n_jobs=n_jobs)
    Sp = sklearn.metrics.pairwise_distances(Rp, metric="euclidean", n_jobs=n_jobs)
    logger.info("Done with RSMs")

    S = double_center(S)
    Sp = double_center(Sp)

    def dCov2(x: npt.NDArray, y: npt.NDArray) -> np.floating:
        return np.multiply(x, y).mean()

    return float(np.sqrt(dCov2(S, Sp) / np.sqrt(dCov2(S, S) * dCov2(Sp, Sp))))


class DistanceCorrelation(RSMSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=distance_correlation,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=True,
        )
