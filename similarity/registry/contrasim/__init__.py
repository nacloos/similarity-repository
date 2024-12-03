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


def compute_cca_squared_correlation(acts1, acts2):
    # copied lines 331-335 from ./contrasim/cca_core.py and added the square inside the mean
    acts1 = acts1.transpose(1, 0)
    acts2 = acts2.transpose(1, 0)
    results = cca_core.get_cca_similarity(acts1, acts2)
    return np.mean(results["cca_coef1"]**2)
similarity.register("contrasim/cca_squared_correlation", compute_cca_squared_correlation)


similarity.register("kernel/contrasim/linear", cka.gram_linear)
similarity.register("kernel/contrasim/rbf-threshold={threshold}", cka.gram_rbf)


def _cosine_similarity(gram_x, gram_y):
    # copied lines 89-95 from ./contrasim/cka.py
    
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

similarity.register("measure/contrasim/cosine", _cosine_similarity)
