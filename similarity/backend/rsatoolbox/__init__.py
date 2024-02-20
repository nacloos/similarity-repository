from functools import partial
import similarity
import rsatoolbox


similarity.register(
    "measure.rsatoolbox",
    {
        "paper_id": "kriegeskorte2008",
        "github": "https://github.com/rsagroup/rsatoolbox"
    }
)

register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d"
    ]
)


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
    # "neg_riem_dist"  # raise error on tests with Gaussians
]

for rdm_method in rdm_methods:
    for compare_method in compare_methods:
        if rdm_method == "poisson" and compare_method == "tau-a":
            # test Gaussians raise ValueError: zero-size array to reduction operation maximum which has no identity
            continue

        compare_method_id = compare_method.replace("-", "_")
        register(
            f"measure.rsatoolbox.rsa-{rdm_method}-{compare_method_id}",
            partial(compute_rsa, rdm_method=rdm_method, compare_method=compare_method)
        )
