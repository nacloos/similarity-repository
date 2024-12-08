"""
https://github.com/google-research/google-research/blob/master/representation_similarity
"""
from functools import partial
import numpy as np

from .demo import *

import similarity


# TODO: register representation_similarity/gram_linear and use standardization?
similarity.register("kernel/representation_similarity/linear", gram_linear)
similarity.register("kernel/representation_similarity/rbf-threshold={threshold}", gram_rbf)
similarity.register("similarity/representation_similarity/centered-cosine", cka)

similarity.register("representation_similarity/cca", cca, preprocessing=["center_columns"])
similarity.register("representation_similarity/cka", feature_space_linear_cka)
similarity.register("representation_similarity/cka_debiased", partial(feature_space_linear_cka, debiased=True))


# copied lines 91-97 from ./contrasim/cka.py
def _cosine_similarity(gram_x, gram_y):    
    # Note: To obtain HSIC, this should be divided by (n-1)**2 (biased variant) or
    # n*(n-3) (unbiased variant), but this cancels for CKA.
    scaled_hsic = gram_x.ravel().dot(gram_y.ravel())

    normalization_x = np.linalg.norm(gram_x)
    normalization_y = np.linalg.norm(gram_y)
    return scaled_hsic / (normalization_x * normalization_y)

similarity.register("measure/representation_similarity/cosine", _cosine_similarity)

