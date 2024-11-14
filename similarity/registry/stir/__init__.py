# https://github.com/nvedant07/STIR
# https://arxiv.org/pdf/2206.11939
from functools import partial
from .stir.CKA_pytorch import linear_CKA, kernel_CKA, linear_HSIC, kernel_HSIC
from .stir.CKA_minibatch import unbiased_linear_HSIC, MinibatchCKA

import similarity



# TODO: how to use minibatch CKA?
# TODO: sigma parameter
sigma = 1.0

similarity.register(
    f"measure/stir/linear_CKA",
    linear_CKA,
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register(
    f"measure/stir/kernel_CKA",
    partial(kernel_CKA, sigma=sigma),
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register(
    f"measure/stir/linear_HSIC",
    linear_HSIC,
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register(
    f"measure/stir/kernel_HSIC",
    partial(kernel_HSIC, sigma=sigma),
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register(
    f"measure/stir/unbiased_linear_HSIC",
    unbiased_linear_HSIC,
    function=True,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

