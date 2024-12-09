import numpy as np

from src.comparators.compare_functions.ps_inv import rearrange_activations


def ls_orth(x1, x2):
    if not x1.shape[0] == x2.shape[0]:
        raise ValueError('Spatial size of compared neurons must match when ' \
                         'calculating psuedo inverse matrix.')

    x1_flat = rearrange_activations(x1)
    x2_flat = rearrange_activations(x2)

    U, S, V = np.linalg.svd(x1_flat.T @ x2_flat)
    w = (U @ V).T
    b = np.zeros(x1_flat.shape[1])

    return {'w': w, 'b': b}
