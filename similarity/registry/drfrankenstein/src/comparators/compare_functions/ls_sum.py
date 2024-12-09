import numpy as np

from src.comparators.compare_functions.ps_inv import rearrange_activations


def truncate(x, rank):
    x = x.copy()
    x[rank:] = 0.
    return x


def ls_sum(x1, x2):
    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating ls_rank matrix.')
    x1 = rearrange_activations(x1)
    x2 = rearrange_activations(x2)
    B = np.linalg.pinv(x1) @ x2
    Y_hat = x1 @ B
    U, S, V = np.linalg.svd(Y_hat, full_matrices=False)
    V = V.T
    matrices = {}
    for rank in [4, 8, 16, len(S)]:
        V_r = V[:, :rank].copy()
        w = (B @ V_r @ V_r.T).T
        b = np.zeros(x1.shape[1])
        matrices[rank] = {'w': w, 'b': b}
    return matrices
