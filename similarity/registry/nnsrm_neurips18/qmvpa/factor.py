""" Helper functions for matrix factorization analysis
"""
import numpy as np
from brainiak.funcalign.srm import SRM
from sklearn.decomposition import PCA


def fit_srm(data_train, data_test, n_components):
    """Fit the shared response model
    Parameters
    ----------
    n_components: k
    data_train: 3d array (n_subj, n_features, n_examples/tps)
    data_test: 3d array (n_subj, n_features, n_examples/tps)

    Returns
    -------
    data_train_sr: 3d array (n_subj, n_components, n_examples/tps)
        the transformed training set
    data_test_sr: 3d array (n_subj, n_components, n_examples/tps)
        the transformed test set
    srm: the fitted model
    """
    assert len(data_train) == len(data_test)
    n_subjects = len(data_train)
    # fit SRM on the training set
    srm = SRM(features=n_components)
    data_train_sr = srm.fit_transform(data_train)
    # transform the hidden activity (on the test set) to the shared space
    data_test_sr = srm.transform(data_test)
    # calculate variance explained
    var_exp_train = calc_srm_var_exp(data_train, data_train_sr, srm.w_)
    return data_train_sr, data_test_sr, srm, var_exp_train

def calc_srm_var_exp(Xs, Xs_sr, Ws): 
    """
    Parameters
    ----------
    Xs: 3d array (n_subj, n_features, n_examples/tps)
    Xs_sr: 3d array (n_subj, n_features, n_examples/tps): the transformed data
    Ws: srm.w_

    Returns
    -------
    var_exp_train [list]: variance explained for each subj 
    """
    n_subjects = len(Xs)
    # calculate total variance for each subject
    subj_var = [np.var(data_subj_i) for data_subj_i in Xs]
    # transform back to native space, then check reconstruction error
    reconstructed = [np.dot(Ws[k], Xs_sr[k]) for k in range(n_subjects)]
    # compute variance explained
    var_exp_train = 1 - np.array([
        np.square(reconstructed[k] - Xs[k]).mean() / subj_var[k]
        for k in range(n_subjects)
    ])
    return var_exp_train
    

def tune_srm(data_train, data_test, n_component_list, var_exp_threshold):
    """
    if you don't care about the var exp curve, this is faster than
    compute_var_exp_srm
    """
    if var_exp_threshold == 1: 
        Xs_train_sr, Xs_test_sr, srm, var_exp_train = fit_srm(
            data_train, data_test, n_component_list[-1])
    else: 
        # fit all srm ...
        for n_component in n_component_list:
            Xs_train_sr, Xs_test_sr, srm, var_exp_train = fit_srm(
                data_train, data_test, n_component)
            # stop if we are happy with var exp
            if var_exp_train > var_exp_threshold:
                break 
    return Xs_train_sr, Xs_test_sr, srm, var_exp_train


def compute_var_exp_srm(data_train, data_test, n_component_list, var_exp_threshold):
    """Trace the test set variance explained curve (but don't double dip!)
        over the number of components
    """
    n_srms = len(n_component_list)
    var_exp_list = np.zeros(n_srms,)
    for i in range(n_srms):
        Xs_train_sr, Xs_test_sr, srm, var_exp_train = fit_srm(
            data_train, data_test, n_component_list[i])
        var_exp_list[i] = var_exp_train
        if var_exp_train > var_exp_threshold:
            final_srm = srm
    return Xs_train_sr, Xs_test_sr, final_srm, var_exp_list


def compute_srm_cost(Xs_sr):
    n_subjs = len(Xs_sr)
    shared_response = np.mean(Xs_sr, axis = 0)
    cost = 0
    for s in range(n_subjs): 
        cost += np.linalg.norm(Xs_sr[s] - shared_response, ord = 'fro')**2
    return cost


def procrustes_align(X_new, S_target):
    """One-step deterministic SRM
    Parameters
    ----------
    X_new:
        an activation trajectory
    S_target: srm_model.s_
        pre-computed shared response as the alignment target

    Returns
    -------
    W:
        the transformation matrix from X to S
    X_aligned:
        the aligned trajectory
    """
    U, s_vals, VT = np.linalg.svd(X_new @ S_target.T, full_matrices=False)
    W = U @ VT
    X_aligned = W.T @ X_new
    return W, X_aligned


def fit_pca(n_components, data_train, data_test):
    """Fit PCA
    Parameters
    ----------
    n_components: k
    data_train: 2d array (n_features, n_examples/tps)
    data_test: 2d array (n_features, n_examples/tps)

    Returns
    -------
    data_train_pca: 3d array (n_subj, n_components, n_examples/tps)
        the transformed training set
    data_test_pca: 3d array (n_components, n_examples/tps)
        the transformed test set
    pca: the fitted model
    """
    pca = PCA(n_components=n_components)
    # tranpose the data to make the format consistent with SRM fit
    data_train_pca = pca.fit_transform(data_train.T).T
    data_test_pca = pca.transform(data_test.T).T
    return data_train_pca, data_test_pca, pca


def fit_pca_thresholded(n_components, var_exp_threshold, data_train, data_test):
    """Fit PCA
    Parameters
    ----------
    n_components: k
    var_exp_threshold: float \in (0,1)
        the amount of variance you wanna explain
    data_train: 2d array (n_features, n_examples/tps)
    data_test: 2d array (n_features, n_examples/tps)

    Returns
    -------
    data_train_pca: 3d array (n_subj, n_components, n_examples/tps)
        the transformed training set
    data_test_pca: 3d array (n_components, n_examples/tps)
        the transformed test set
    pca: the fitted model
    """
    # fit PCA with the input number of components
    _, _, pca_model = fit_pca(n_components, data_train, data_test)
    # compute prop variance explained by the 1st k components
    cum_var_exp = np.cumsum(pca_model.explained_variance_ratio_)
    # find k, s.t. PCA(k) explain more variance than the threshold
    candidiates_components = np.where(cum_var_exp > var_exp_threshold)[0]
    # choose k = 1st candidate or the input components
    if len(candidiates_components) == 0:
        n_components_threshold = n_components
    else:
        n_components_threshold = candidiates_components[0]
    # re-fit PCA
    data_train_pca, data_test_pca, final_pca_model = fit_pca(
        n_components_threshold, data_train, data_test)
    return data_train_pca, data_test_pca, final_pca_model


"""utils
"""


def chose_n_components(n_component_list, cum_var_exp, var_exp_threshold):
    assert len(n_component_list) == len(cum_var_exp)
    assert var_exp_threshold > 0 and var_exp_threshold < 1
    # find k, s.t. PCA(k) explain more variance than the threshold
    candidiates_ids = np.where(cum_var_exp > var_exp_threshold)[0]
    # choose k = 1st candidate or the input components
    if len(candidiates_ids) == 0:
        best_n_component = n_component_list[-1]
    else:
        best_n_component = n_component_list[candidiates_ids[0]]
    return best_n_component
