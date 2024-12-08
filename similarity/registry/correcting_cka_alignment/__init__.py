# https://github.com/Alxmrphi/correcting_CKA_alignment
from functools import partial

import numpy as np
from .metrics import cka, gram_linear

import similarity


def _measure(X, Y, debiased=False):
    X_gram = gram_linear(X)
    Y_gram = gram_linear(Y)
    return cka(X_gram, Y_gram, debiased=debiased)


for debiased in [True, False]:
    name = "cka"
    if debiased:
        name += "_debiased"

    similarity.register(
        f"correcting_cka_alignment/{name}",
        partial(_measure, debiased=debiased),
    )


similarity.register("kernel/correcting_cka_alignment/linear", gram_linear)


def _cosine_similarity(gram_x, gram_y):
    # copy pasted lines 31-37 from ./metrics.py
    
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

similarity.register("measure/correcting_cka_alignment/cosine", _cosine_similarity)
