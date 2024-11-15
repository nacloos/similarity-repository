# https://github.com/technion-cs-nlp/ContraSim
# https://arxiv.org/pdf/2303.16992
from functools import partial

import numpy as np
from .contrasim import cka, pwcca, cca_core

import similarity


similarity.register("contrasim/feature_space_linear_cka", cka.feature_space_linear_cka)
similarity.register("contrasim/feature_space_linear_cka_debiased", partial(cka.feature_space_linear_cka, debiased=True))
similarity.register("contrasim/pwcca", pwcca.compute_pwcca)
similarity.register("contrasim/cca", cca_core.compute_cca)
similarity.register("contrasim/svcca", cca_core.compute_svcca)

similarity.register("kernel/contrasim/linear", cka.gram_linear)

def _cosine_similarity(gram_x, gram_y):
    # copy pasted lines 89-95 from ./contrasim/cka.py
    
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

similarity.register("measure/contrasim/cosine", _cosine_similarity)
