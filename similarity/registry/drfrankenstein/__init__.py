# https://github.com/renyi-ai/drfrankenstein
from .src.comparators.compare_functions import cca, cka, correlation, l2, lr

import similarity


similarity.register(
    "measure/drfrankenstein/cca",
    cca,
    function=True
)

similarity.register(
    "measure/drfrankenstein/cka",
    cka,
    function=True
)


similarity.register(
    "measure/drfrankenstein/l2",
    l2,
    function=True
)

similarity.register(
    "measure/drfrankenstein/lr",
    lr,
    function=True
)
