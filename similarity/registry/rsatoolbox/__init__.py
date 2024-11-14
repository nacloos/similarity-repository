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
    sim = sim[0][0]
    return sim


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


# similarity.register(
#     "measure/rsatoolbox",
#     {
#         "paper_id": "kriegeskorte2008",
#         "github": "https://github.com/rsagroup/rsatoolbox"
#     }
# )

# register = partial(
#     similarity.register, 
#     function=True,
#     preprocessing=[
#         "reshape2d"
#     ]
# )


# def centering(rdm):
#     X = rdm.get_matrices()[0]
#     n = X.shape[0]
#     C = np.eye(n) - np.ones([n, n]) / n
#     X_centered = C @ X @ C
#     rdm = rsatoolbox.rdm.rdms.RDMs(
#         dissimilarities=np.array([X_centered]),
#         dissimilarity_measure=rdm.dissimilarity_measure,
#         descriptors=rdm.descriptors,
#         rdm_descriptors=rdm.rdm_descriptors,
#         pattern_descriptors=rdm.pattern_descriptors
#     )
#     return rdm


# def compute_rsa(X, Y, rdm_method, compare_method, center_rdm=False):
#     """
#     Helper function to compute representational similarity analysis (RSA) using the rsatoolbox library
#     """
#     X = rsatoolbox.data.Dataset(X)
#     Y = rsatoolbox.data.Dataset(Y)

#     rdm1 = rsatoolbox.rdm.calc_rdm(X, method=rdm_method)
#     rdm2 = rsatoolbox.rdm.calc_rdm(Y, method=rdm_method)

#     # TODO: the diagonal is removed when rsatoolbox creates RDMs objects
#     # so the result is different from centered cosine similarity of sq eucl RDMs

#     # add the option for centering the RDMs for the equivalence with CKA (see https://openreview.net/pdf?id=zMdnnFasgC)
#     if center_rdm:
#         rdm1 = centering(rdm1)
#         rdm2 = centering(rdm2)

#     sim = rsatoolbox.rdm.compare(rdm1, rdm2, method=compare_method)
#     sim = sim[0][0]
#     return sim


# rdm_methods = [
#     "euclidean",
#     "correlation", 
#     "mahalanobis",
#     # "crossnobis",  # raise ValueError: descriptor must be a string! Crossvalidationrequires multiple measurements to be grouped
#     "poisson",
#     # "poisson_cv"  # raise ValueError: descriptor must be a string! Crossvalidationrequires multiple measurements to be grouped
# ]

# compare_methods = [
#     "cosine",
#     "spearman",
#     "corr",
#     "kendall", 
#     "tau-b",
#     "tau-a",
#     "rho-a",
#     "corr_cov",
#     "cosine_cov",
#     # "neg_riem_dist",  # raise error on tests with Gaussians
#     "bures",
#     "bures_metric",
# ]

# for rdm_method in rdm_methods:
#     for compare_method in compare_methods:
#         for center_rdm in [True, False]:
#             if rdm_method == "poisson" and compare_method == "tau-a":
#                 # test Gaussians raise ValueError: zero-size array to reduction operation maximum which has no identity
#                 continue

#             compare_method_id = compare_method.replace("-", "_")
            
#             if center_rdm:
#                 name = f"rsa-{rdm_method}_centered_rdm-{compare_method_id}"
#             else:
#                 name = f"rsa-{rdm_method}_rdm-{compare_method_id}"

#             register(
#                 f"measure/rsatoolbox/{name}",
#                 partial(compute_rsa, rdm_method=rdm_method, compare_method=compare_method, center_rdm=center_rdm)
#             )
