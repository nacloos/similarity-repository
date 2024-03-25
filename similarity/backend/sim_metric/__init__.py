from functools import partial

import similarity
from . import utils
from .dists import scoring


similarity.register(
    "measure.sim_metric",
    {
        "paper_id": "ding2021",
        "github": "https://github.com/js-d/sim_metric"
    }
)

register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d",
        "center_columns",
        # sim_metric scoring functions expect shape (neuron, sample)
        # but measure inputs are of shape (sample, neuron)
        "transpose"
    ]
)

register(
    "measure.sim_metric.cca",
    utils.mean_cca_corr
)
register(
    "measure.sim_metric.cca-mean_sq_corr",
    utils.mean_sq_cca_corr,
)
register(
    "measure.sim_metric.pwcca",
    utils.pwcca_dist
)
register(
    "measure.sim_metric.cka",
    scoring.lin_cka_dist,
    postprocessing=[
        # (Ding, 2021): d_CKA = 1 - CKA
        "one_minus"
    ]
)
register(
    "measure.sim_metric.procrustes-sq-euclidean",
    scoring.procrustes
)
