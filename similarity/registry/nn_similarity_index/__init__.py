from functools import partial

import similarity
from . import utils


# similarity.register(
#     "measure/nn_similarity_index",
#     {
#         "paper_id": "tang2020",
#         "github": "https://github.com/amzn/xfer/blob/master/nn_similarity_index"
#     }
# )


similarity.register("nn_similarity_index/euclidean", utils.euclidean)
similarity.register("nn_similarity_index/cka", utils.cka)
similarity.register("nn_similarity_index/nbs", utils.nbs)
similarity.register("nn_similarity_index/bures_distance", utils.bures_distance)
