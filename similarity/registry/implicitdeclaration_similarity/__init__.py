# https://github.com/implicitDeclaration/similarity
from functools import partial
from .metric.CKA import linear_CKA, sparse_HSIC

import similarity


similarity.register(
    "implicitdeclaration_similarity/linear_cka",
    linear_CKA,
    function=True
)


# TODO
topk = 500
sigma = 1.0
kernels = ["linear", "rbf", "cos"]
for kernel in kernels:
    name = f"{kernel}_sparse_HSIC"
    similarity.register(
        f"measure/implicitdeclaration_similarity/{name}",
        partial(sparse_HSIC, topk=topk, kernel=kernel, sigma=sigma),
        function=True
    )

