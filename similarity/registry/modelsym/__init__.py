# https://github.com/pnnl/modelsym
from functools import partial

from .model_symmetries.alignment.alignment import (
    wreath_cka,
    wreath_procrustes,
    ortho_cka,
    ortho_procrustes
)

import similarity


register = partial(
    similarity.register,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

register("modelsym/wreath_cka", lambda X, Y: wreath_cka([X[None]], [Y[None]]))
register("modelsym/wreath_procrustes", lambda X, Y: wreath_procrustes([X[None]], [Y[None]]))
register("modelsym/ortho_cka", lambda X, Y: ortho_cka([X[None]], [Y[None]]))
register("modelsym/ortho_procrustes", lambda X, Y: ortho_procrustes([X[None]], [Y[None]]))
