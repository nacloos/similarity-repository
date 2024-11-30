from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.spatial.distance
import torch

from llmcomp.measures.utils import to_numpy_if_needed


def rsm_norm_diff(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    inner: str = "euclidean",
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    S = scipy.spatial.distance.pdist(R, inner)  # type:ignore
    Sp = scipy.spatial.distance.pdist(Rp, inner)  # type:ignore
    return float(
        np.linalg.norm(S - Sp, ord=2)
    )  # ord=2 because pdist gives vectorized lower triangle of RSM
