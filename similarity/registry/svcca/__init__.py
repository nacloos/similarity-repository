from functools import partial

import numpy as np

import similarity

from . import cca_core, pwcca, numpy_pls


similarity.register(
    "paper/svcca",
    {
        "id": ["raghu2017", "morcos2018"],
        "github": "https://github.com/google/svcca"
    }
)


register = partial(
    similarity.register,
    preprocessing=[
        "reshape2d",
        # svcca repo's functions expect data with shape (neuron, data_point)
        # and similarity.measure expects data with shape (data_point, neuron)
        "transpose"
    ]
)

register(
    "svcca/cca",
    partial(cca_core.get_cca_similarity, verbose=False),
    postprocessing=[
        # get_cca_similarity returns a dict and the value for "mean" is a tuple of len 2 with twice the same value
        # lambda score: score["mean"][0]
        lambda score: np.mean(score["cca_coef1"])
    ]
)
register(
    "svcca/cca_squared_correlation",
    partial(cca_core.get_cca_similarity, verbose=False),
    postprocessing=[
        lambda score: np.mean(score["cca_coef1"]**2)
    ]
)
register(
    "svcca/pwcca",
    pwcca.compute_pwcca,
    postprocessing=[
        # use only the mean, which is the first output of compute_pwcca
        lambda score: score[0]
    ]
)
register(
    "svcca/pls",
    numpy_pls.get_pls_similarity,
    postprocessing=[
        # modified get_pls_similarity to return the mean of the eigenvalues, used as similarity score here
        lambda score: score["mean"]
    ]
)