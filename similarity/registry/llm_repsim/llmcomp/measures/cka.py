from typing import Union

import numpy.typing as npt
import torch

from llmcomp.measures.utils import to_torch_if_needed


def centered_kernel_alignment(
    R: Union[torch.Tensor, npt.NDArray], Rp: Union[torch.Tensor, npt.NDArray]
) -> float:
    """Kornblith et al. (2019)"""
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
            / (
                torch.linalg.norm(R.T @ R, ord="fro")
                * torch.linalg.norm(Rp.T @ Rp, ord="fro")
            )
        ).item()


def hsic(S: torch.Tensor, Sp: torch.Tensor) -> torch.Tensor:  # noqa: E741
    S = S - S.mean(dim=0)[:, None]
    Sp = Sp - Sp.mean(dim=0)[:, None]  # noqa: E741
    return torch.trace(S @ Sp) / (S.size(0) - 1) ** 2
