from typing import List

import torch

from . import CKA_pytorch as CKA_pytorch


def unbiased_linear_HSIC(X: torch.Tensor, Y: torch.Tensor) -> torch.Tensor:
    """
    Computes the unbised HSIC estimator for activations X and Y.

    See https://arxiv.org/pdf/2010.15327.pdf equation 3
    Partially inspired by:
    https://github.com/AntixK/PyTorch-Model-Compare/blob/main/torch_cka/cka.py

    Parameters
    ----------
    X : torch.Tensor
        First set of activations.
    Y : torch.Tensor
        Second set of activations.
    """
    K = CKA_pytorch.centering(X @ X.T).fill_diagonal_(0)
    L = CKA_pytorch.centering(Y @ Y.T).fill_diagonal_(0)

    n_samples = K.size(0)
    ones = torch.ones(n_samples, 1, device=K.device, dtype=torch.float32)
    return (
        (
            torch.trace(K @ L)
            + (
                (ones.T @ K @ ones @ ones.T @ L @ ones)
                / ((n_samples - 1) * (n_samples - 2))
            )
            - 2 * (ones.T @ K @ L @ ones) / (n_samples - 2)
        ) / n_samples * (n_samples - 3)
    )


class MinibatchCKA:
    """
    Computes minibatch-CKA aggregated over multiple batches of data.

    For each minibatch and its corresponding set of representations
    X_i, Y_i, call .add_minibatch(X_i, Y_i).
    To obtain the CKA-value aggregated over all .add_minibatch() calls
    since the last time .reset() was called, use .value().
    The aggregator state is resetted when calling .reset()
    or .value(reset=True),

    See tests/test_CKA_minibatch.py for usage examples.
    """
    def __init__(self) -> None:
        self.reset()

    def reset(self) -> None:
        self.hsic_xy: List[torch.Tensor] = []
        self.hsic_xx: List[torch.Tensor] = []
        self.hsic_yy: List[torch.Tensor] = []

    def add_minibatch(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        self.hsic_xy.append(unbiased_linear_HSIC(X, Y).float().detach().cpu())
        self.hsic_xx.append(unbiased_linear_HSIC(X, X).float().detach().cpu())
        self.hsic_yy.append(unbiased_linear_HSIC(Y, Y).float().detach().cpu())

    def __call__(self, X: torch.Tensor, Y: torch.Tensor) -> None:
        return self.add_minibatch(X, Y)

    def value(self, reset: bool = False) -> torch.Tensor:
        """
        See https://arxiv.org/pdf/2010.15327.pdf equation 2
        """
        n_batches = len(self.hsic_xy)
        agg_hsic = sum(self.hsic_xy) / n_batches
        agg_var1 = torch.sqrt(torch.div(sum(self.hsic_xx), n_batches))
        agg_var2 = torch.sqrt(torch.div(sum(self.hsic_yy), n_batches))

        if reset:
            self.reset()
        return agg_hsic / (agg_var1 * agg_var2)
