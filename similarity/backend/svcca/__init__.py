from functools import partial

import similarity

from . import cca_core, pwcca, numpy_pls


similarity.register(
    "measure.svcca",
    {
        "paper_id": "raghu2017",
        "github": "https://github.com/google/svcca"
    }
)


register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d",
        # svcca repo's functions expect data with shape (neuron, data_point)
        # and similarity.measure expects data with shape (data_point, neuron)
        "transpose"
    ]
)

register(
    "measure.svcca.cca",
    partial(cca_core.get_cca_similarity, verbose=False),
    postprocessing=[
        # get_cca_similarity returns a dict and the value for "mean" is a tuple of len 2 with twice the same value
        lambda score: score["mean"][0]
    ]
)
register(
    "measure.svcca.pwcca",
    pwcca.compute_pwcca,
    postprocessing=[
        # use only the mean, which is the first output of compute_pwcca
        lambda score: score[0]
    ]
)
register(
    "measure.svcca.pls",
    numpy_pls.get_pls_similarity,
    postprocessing=[
        # modified get_pls_similarity to return the mean of the eigenvalues, used as similarity score here
        lambda score: score["mean"]
    ]
)
