import numpy as np
from scipy.linalg import orthogonal_procrustes


def procrustes(X, Y):
    # TODO
    r, scale = orthogonal_procrustes(X, Y)
    total_norm = (
        -2 * scale
        + np.linalg.norm(X, ord="fro") ** 2
        + np.linalg.norm(Y, ord="fro") ** 2
    )
    return total_norm
