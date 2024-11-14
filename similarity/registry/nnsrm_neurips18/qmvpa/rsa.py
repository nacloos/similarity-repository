"""Functions for Representational Similarity Analysis
"""

import numpy as np
from scipy.stats import pearsonr
from scipy.spatial import procrustes
from .utils import reflect_upper_triangular_part, \
    vectorize_lower_trigular_part


def within_RSMs(Xs):
    """Compute all within-subject RSMs
    Parameters
    ----------
    Xs: a list of 2d array in the form of (n_feature_dim x n_examples)
        the activation matrices

    Returns
    -------
    rsms
        a list of representational similarity matrices
    """
    # compute RSM
    num_subjects = len(Xs)
    rsms = [inter_RSM(Xs[s], Xs[s]) for s in range(num_subjects)]
    return rsms


def correlate_2RSMs(rsm1, rsm2):
    """Compute the correlation between 2 RSMs (2nd order correlations)
    Parameters
    ----------
    rsm_i: a 2d array in the form of (n_examples x n_examples)
        a representational similarity matrix

    Returns
    -------
    r: float
        linear_correlation(rsm1, rsm2)
    """
    assert np.shape(rsm1) == np.shape(rsm2)
    # only compare the lower triangular parts (w/o diagonal values)
    rsm1_vec_lower = vectorize_lower_trigular_part(rsm1)
    rsm2_vec_lower = vectorize_lower_trigular_part(rsm2)
    # r = np.corrcoef(rsm1_vec_lower, rsm2_vec_lower)[0, 1]
    r_val, p_val = pearsonr(rsm1_vec_lower, rsm2_vec_lower)
    return r_val, p_val


def correlate_RSMs(rsms):
    """Compute correlation between RSMs (2nd order correlations)
    Parameters
    ----------
    rsms: a list of 2d array in the form of (n_examples x n_examples)
        representational similarity matrces

    Returns
    -------
    rsm_corrs: 2d array in the form of (num_rsms x num_rsms)
    """
    num_subjects = len(rsms)
    rsm_corrs = np.zeros((num_subjects, num_subjects))
    rsm_ps = np.zeros((num_subjects, num_subjects))
    for i in range(num_subjects):
        for j in np.arange(0, i+1, 1):
            rsm_corrs[i, j], rsm_ps[i, j] = correlate_2RSMs(
                rsms[i], rsms[j])
    # fillin the upper triangular part by symmetry
    rsm_corrs = reflect_upper_triangular_part(rsm_corrs)
    rsm_ps = reflect_upper_triangular_part(rsm_ps)
    return rsm_corrs, rsm_ps


def inter_RSM(m1, m2):
    """Compute the RSM for 2 activation matrices
    Parameters
    ----------
    mi: 2d array (n_feature_dim x n_examples)
        a activation matrix

    Returns
    -------
    intersubj_rsm
        corr(col_i(m1), col_j(m2)), for all i and j
    """
    assert np.shape(m1) == np.shape(m2)
    n_examples = np.shape(m1)[1]
    intersubj_rsm = np.corrcoef(m1.T, m2.T)[:n_examples, n_examples:]
    return intersubj_rsm


def inter_RSMs(matrix_list):
    """Compute intersubject representational similarity for a list of acts
        when comparing (k-1) subjects to left-out subject, average the k

    Parameters
    ----------
    matrix_list: a list of 2d array in the form of (n_feature_dim x n_examples)
        the activation matrices

    Returns
    -------
    intersubj_rsm
        the average intersubject representational similarity matrix
    """
    matrix_array = np.array(matrix_list)
    len(matrix_list)
    intersubj_rsms = []
    for loo_idx in range(len(matrix_list)):
        mean_Hs = np.mean(
            matrix_array[np.arange(len(matrix_list)) != loo_idx], axis=0)
        intersubj_rsms.append(inter_RSM(matrix_array[loo_idx], mean_Hs))
    intersubj_rsm = np.mean(intersubj_rsms, axis=0)
    return intersubj_rsm


def inter_procrustes(matrix_array):
    # input: matrix_array, n_subj x n_units x n_examples
    n_nets = np.shape(matrix_array)[0]
    D = np.zeros((n_nets, n_nets))
    for i in range(n_nets):
        for j in np.arange(0, i):
            _, _, D[i, j] = procrustes(matrix_array[i], matrix_array[j])
    return D


def isc(X_i, X_j):
    """Compute ISC across 2 subjecsts: subject i <-> subject j
    Parameters
    ----------
    X_i: 2d array - num_voxels_i x num_examples
    X_j: 2d array - num_voxels_j x num_examples
        the activation matrices

    Returns
    -------
    isc_ij: 2d array - num_voxels_i x num_voxels_j
    isc_ii: 2d array - num_voxels_i x num_voxels_i
    isc_jj: 2d array - num_voxels_j x num_voxels_j
    """
    assert np.shape(X_i)[1] == np.shape(X_j)[1]
    # compute the full isc map
    # with shape (num_voxels_i + num_voxels_j) x (num_voxels_i + num_voxels_j)
    isc_map = np.corrcoef(X_i, X_j)
    # divide isc maps
    n_voxs_i = np.shape(X_i)[0]
    isc_ij = isc_map[:n_voxs_i, n_voxs_i:]
    isc_ii = isc_map[:n_voxs_i, :n_voxs_i]
    isc_jj = isc_map[n_voxs_i:, n_voxs_i:]
    return isc_ij, isc_ii, isc_jj


def isc_pairwise(Xs):
    """Compute all pair-wise ISC
    Parameters
    ----------
    Xs: {X_i : X_i is num_voxels_i x num_examples}

    Returns
    -------
    isc_list: the list of isc matrices for subj i and subj j for i != j
    isc_ij_mean: the mean of the list above
    ij_indices: the i-j pairs ordering
    """
    ij_indices = []
    isc_list = []
    isc_ij_mean = np.zeros((len(Xs), len(Xs)))
    for i in range(len(Xs)):
        for j in range(i):
            # get inter-isc
            isc_ij, _, _ = isc(Xs[i], Xs[j])
            # record the computations
            ij_indices.append([i, j])
            isc_list.append(isc_ij)
            isc_ij_mean[i, j] = np.mean(isc_ij)
    return isc_list, isc_ij_mean, ij_indices
