from typing import Union

import numpy.typing as npt
import torch
from resi.measures.utils import flatten
from resi.measures.utils import RepresentationalSimilarityMeasure
from resi.measures.utils import SHAPE_TYPE
from resi.measures.utils import to_torch_if_needed


def centered_kernel_alignment(
    R: Union[torch.Tensor, npt.NDArray],
    Rp: Union[torch.Tensor, npt.NDArray],
    shape: SHAPE_TYPE,
) -> float:
    """Kornblith et al. (2019)"""
    R, Rp = flatten(R, Rp, shape=shape)
    R, Rp = to_torch_if_needed(R, Rp)
    N, D = R.size()

    R = R - R.mean(dim=0)[None, :]
    Rp = Rp - Rp.mean(dim=0)[None, :]

    if N < D:
        S = R @ R.T
        Sp = Rp @ Rp.T  # noqa: E741
        return (hsic(S, Sp) / torch.sqrt(hsic(S, S) * hsic(Sp, Sp))).item()
    else:
        return (
            torch.linalg.norm(Rp.T @ R, ord="fro") ** 2
            / (torch.linalg.norm(R.T @ R, ord="fro") * torch.linalg.norm(Rp.T @ Rp, ord="fro"))
        ).item()


def hsic(S: torch.Tensor, Sp: torch.Tensor) -> torch.Tensor:  # noqa: E741
    S = S - S.mean(dim=0)[:, None]
    Sp = Sp - Sp.mean(dim=0)[:, None]  # noqa: E741
    return torch.trace(S @ Sp) / (S.size(0) - 1) ** 2


class CKA(RepresentationalSimilarityMeasure):
    def __init__(self):
        super().__init__(
            sim_func=centered_kernel_alignment,
            larger_is_more_similar=True,
            is_metric=False,
            is_symmetric=True,
            invariant_to_affine=False,
            invariant_to_invertible_linear=False,
            invariant_to_ortho=True,
            invariant_to_permutation=True,
            invariant_to_isotropic_scaling=True,
            invariant_to_translation=True,
        )
