from functools import partial
from . import utils
from .dists import scoring

from similarity import register

# TODO: name backend by author last name and date instead of github repo name?

register = partial(
    register,
    function=True,
    preprocessing=[
        "reshape2d",
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
    scoring.lin_cka_dist
)
register(
    "measure.sim_metric.procrustes-sq-euclidean",
    scoring.procrustes
)
