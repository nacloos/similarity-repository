# https://github.com/mgwillia/unsupervised-analysis
# https://arxiv.org/pdf/2206.08347
from .experiments.calculate_cka import CudaCKA

import similarity

similarity.register(
    "unsupervised_analysis/cka",
    lambda X, Y: CudaCKA(device="cpu").linear_CKA(X, Y),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
