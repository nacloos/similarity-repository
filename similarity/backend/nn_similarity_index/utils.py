from .sim_indices import SimIndex


def compute_kernels(X, Y):
    # compute kernel matrix according to eq (6) of the paper (https://arxiv.org/pdf/2003.11498.pdf)
    # X, Y here have shape (sample, neuron) (the transpose of the paper's notation)
    return X @ X.T, Y @ Y.T


# small interface to be able to use the class methods as functions
def euclidean(kmat_1, kmat_2):
    return SimIndex().euclidean(kmat_1, kmat_2)


def cka(kmat_1, kmat_2):
    return SimIndex().cka(kmat_1, kmat_2)


def nbs(kmat_1, kmat_2):
    return SimIndex().nbs(kmat_1, kmat_2)
