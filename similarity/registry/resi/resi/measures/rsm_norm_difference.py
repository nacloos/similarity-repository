from typing import Optional
from typing import Union

import numpy as np
import numpy.typing as npt
import sklearn.metrics
import torch
from resi.measures.utils import flatten
from resi.measures.utils import RSMSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_numpy_if_needed


def rsm_norm_diff(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
    inner: str = "euclidean",
    n_jobs: Optional[int] = None,
) -> float:
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_numpy_if_needed(R, Rp)
    S = sklearn.metrics.pairwise_distances(R, metric=inner, n_jobs=n_jobs)  # type:ignore
    Sp = sklearn.metrics.pairwise_distances(Rp, metric=inner, n_jobs=n_jobs)  # type:ignore
    return float(np.linalg.norm(S - Sp, ord=2))  # ord=2 because pdist gives vectorized lower triangle of RSM


class RSMNormDifference(RSMSimilarityMeasure):
    def __init__(self):
        # inner is fixed to euclidean, so these value should not be changed
        super().__init__(
            sim_func=rsm_norm_diff,
            larger_is_more_similar=False,
            is_metric=True,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=False,
            invariant_to_translation=False,
        )

    def __call__(
        self,
        R: torch.Tensor | npt.NDArray,
        Rp: torch.Tensor | npt.NDArray,
        shape: SHAPE_TYPE,
        inner: str = "euclidean",
        n_jobs: int = 1,
    ) -> float:
        return self.sim_func(R, Rp, shape, inner=inner, n_jobs=n_jobs)  # type:ignore
