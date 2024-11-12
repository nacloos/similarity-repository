from functools import partial

import similarity
from . import utils


similarity.register(
    "measure.nn_similarity_index",
    {
        "paper_id": "tang2020",
        "github": "https://github.com/amzn/xfer/blob/master/nn_similarity_index"
    }
)


register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d",
        "center_columns",
    ]
)

register(
    # euclidean distance between kernel matrices
    "measure.nn_similarity_index.rsa-correlation-euclidean",
    utils.euclidean
)
register(
    "measure.nn_similarity_index.cka",
    utils.cka
)
register(
    "measure.nn_similarity_index.nbs",
    utils.nbs
)
register(
    "measure.nn_similarity_index.bures_distance",
    utils.bures_distance
)
