"""
Small interface to be able to use the class methods as functions.
This code is not part of the original 'nn_similarity_index' repository.
"""
import numpy as np
from .sim_indices import SimIndex


def compute_kernel(X):
    # compute kernel matrix according to eq (6) of the paper (https://arxiv.org/pdf/2003.11498.pdf)
    # X here has shape (sample, neuron) (the transpose of the paper's notation)
    return X @ X.T


def euclidean(kmat_1, kmat_2):
    return SimIndex().euclidean(kmat_1, kmat_2)


def cka(kmat_1, kmat_2):
    return SimIndex().cka(kmat_1, kmat_2)


def nbs(kmat_1, kmat_2):
    return SimIndex().nbs(kmat_1, kmat_2)


# this metric is not included in the original code but added for comparison purposes
def bures_distance(kmat_1, kmat_2):
    kmat_1 = SimIndex().centering(kmat_1)
    kmat_2 = SimIndex().centering(kmat_2)
    # simple rearrangement of the terms in the SimIndex.nbs method
    return (np.trace(kmat_1) + np.trace(kmat_2) - 2 * sum(np.real(np.linalg.eigvals(kmat_1 @ kmat_2)).clip(0.) ** 0.5)) ** 0.5
