from functools import partial
import numpy as np

import rsatoolbox

import similarity


def compute_rsa(X, Y, rdm_method, compare_method):
    """
    Helper function to compute representational similarity analysis (RSA) using the rsatoolbox library
    """
    X = rsatoolbox.data.Dataset(X)
    Y = rsatoolbox.data.Dataset(Y)

    rdm1 = rsatoolbox.rdm.calc_rdm(X, method=rdm_method)
    rdm2 = rsatoolbox.rdm.calc_rdm(Y, method=rdm_method)

    sim = rsatoolbox.rdm.compare(rdm1, rdm2, method=compare_method)
    return sim[0][0]


rdm_methods = [
    "euclidean",
    "correlation", 
    "mahalanobis",
    # "crossnobis",  # raise ValueError: descriptor must be a string! Crossvalidationrequires multiple measurements to be grouped
    "poisson",
    # "poisson_cv"  # raise ValueError: descriptor must be a string! Crossvalidationrequires multiple measurements to be grouped
]

compare_methods = [
    "cosine",
    "spearman",
    "corr",
    "kendall", 
    "tau-b",
    "tau-a",
    "rho-a",
    "corr_cov",
    "cosine_cov",
    # "neg_riem_dist",  # raise error on tests with Gaussians
    "bures",
    "bures_metric",
]

for rdm_method in rdm_methods:
    for compare_method in compare_methods:
        if rdm_method == "poisson" and compare_method == "tau-a":
            # test Gaussians raise ValueError: zero-size array to reduction operation maximum which has no identity
            continue

        compare_method_id = compare_method.replace("-", "_")
        
        similarity.register(
            f"rsatoolbox/rsa-{rdm_method}-{compare_method_id}",
            partial(compute_rsa, rdm_method=rdm_method, compare_method=compare_method)
        )


# register RDM methods
def _compute_rdm(X, method):
    dataset = rsatoolbox.data.Dataset(X)
    rdm = rsatoolbox.rdm.calc_rdm(dataset, method=method)
    return rdm.get_matrices()[0]


# rsatoolbox removes the diagonal and normalizes the RDM
# rdm = _extract_triu_(Dx) / X.shape[1]
similarity.register(
    "rdm/rsatoolbox/squared_euclidean_normalized",
    partial(_compute_rdm, method="euclidean")
)

similarity.register(
    "rdm/rsatoolbox/squared_mahalanobis_normalized",
    partial(_compute_rdm, method="mahalanobis")
)


# register RDM comparison methods
def _compare_rdm(rdm1: np.ndarray, rdm2: np.ndarray, method: str):
    # convert 2D numpy arrays to rsatoolbox RDMs objects
    assert rdm1.ndim == 2
    assert rdm2.ndim == 2

    # rsatoolbox requires square matrices
    # pad with zeros if not square but this function shouldn't be used for non-square matrices
    if rdm1.shape[0] != rdm1.shape[1]:
        rdm1 = np.pad(rdm1, ((0, 0), (0, rdm1.shape[0] - rdm1.shape[1])), mode="constant")
    if rdm2.shape[0] != rdm2.shape[1]:
        rdm2 = np.pad(rdm2, ((0, 0), (0, rdm2.shape[0] - rdm2.shape[1])), mode="constant")

    rdm1 = rsatoolbox.rdm.RDMs(rdm1[None])
    rdm2 = rsatoolbox.rdm.RDMs(rdm2[None])
    sim = rsatoolbox.rdm.compare(rdm1, rdm2, method=method)
    return sim[0][0]

similarity.register(
    "measure/rsatoolbox/zero_diagonal-cosine",
    partial(_compare_rdm, method="cosine")
)

similarity.register(
    "similarity/rsatoolbox/zero_diagonal-bures",
    partial(_compare_rdm, method="bures")
)
