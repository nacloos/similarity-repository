# https://github.com/Alxmrphi/correcting_CKA_alignment
from functools import partial
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
        function=True
    )


