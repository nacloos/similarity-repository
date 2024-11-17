from functools import partial

import similarity
from . import utils
from .dists import scoring


similarity.register(
    "paper/sim_metric",
    {
        "id": "ding2021",
        "github": "https://github.com/js-d/sim_metric"
    }
)

register = partial(
    similarity.register,
    preprocessing=[
        "center_columns",
        # sim_metric scoring functions expect shape (neuron, sample)
        # but measure inputs are of shape (sample, neuron)
        "transpose"
    ]
)

register("sim_metric/mean_cca_corr", utils.mean_cca_corr)
register("sim_metric/mean_sq_cca_corr", utils.mean_sq_cca_corr)
register("sim_metric/pwcca_dist", utils.pwcca_dist)
register("sim_metric/lin_cka_dist", scoring.lin_cka_dist)
register("sim_metric/procrustes", scoring.procrustes)
