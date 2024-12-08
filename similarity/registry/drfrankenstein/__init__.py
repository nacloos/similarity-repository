# https://github.com/renyi-ai/drfrankenstein
from functools import partial
from .src.comparators.compare_functions import cca, cka, correlation, l2, lr

import similarity


similarity.register("drfrankenstein/cca", cca, preprocessing=["center_columns"])
similarity.register("drfrankenstein/cka", cka)


# TODO: register distance
# similarity.register(
#     "drfrankenstein/l2",
#     l2,
# )

# similarity.register(
#     "drfrankenstein/lr",
#     lr,
# )
