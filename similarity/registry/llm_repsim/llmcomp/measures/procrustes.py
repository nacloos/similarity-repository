from typing import Union

import numpy as np
import numpy.typing as npt
import scipy.linalg
import scipy.optimize
import torch

from llmcomp.measures.utils import adjust_dimensionality, to_numpy_if_needed


def orthogonal_procrustes(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    nucnorm = scipy.linalg.orthogonal_procrustes(R, Rp)[1]
    return np.sqrt(
        -2 * nucnorm
        + np.linalg.norm(R, ord="fro") ** 2
        + np.linalg.norm(Rp, ord="fro") ** 2
    )


def aligned_cossim(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    R, Rp = to_numpy_if_needed(R, Rp)
    R, Rp = adjust_dimensionality(R, Rp)
    align, _ = scipy.linalg.orthogonal_procrustes(R, Rp)

    R_aligned = R @ align
    sum_cossim = 0
    for r, rp in zip(R_aligned, Rp):
        sum_cossim += r.dot(rp) / (np.linalg.norm(r) * np.linalg.norm(rp))
    return sum_cossim / R.shape[0]
