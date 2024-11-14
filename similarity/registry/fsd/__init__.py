# https://github.com/maroo-sky/FSD
from collections import namedtuple
from functools import partial

from .metrics.LayerWiseMetrics import linear_HSIC, linear_CKA_loss

import similarity

# TODO: what is RKD?

args = namedtuple('Args', ['device'])('cpu')


similarity.register(
    "measure/fsd/linear_CKA_loss",
    partial(linear_CKA_loss, args=args),
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)


similarity.register(
    "measure/fsd/linear_HSIC",
    partial(linear_HSIC, args=args),
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
