# https://github.com/mtoneva/brain_language_nlp
# https://arxiv.org/pdf/1905.11833
from functools import partial
import numpy as np
from .utils import ridge_tools

import similarity


def wrap_ridge_with_r2(ridge_func, X, Y, lmbda=1):
    weights = ridge_func(X, Y, lmbda)
    predictions = np.dot(X, weights)
    # TODO: ok to return the mean R2?
    return np.mean(ridge_tools.R2(predictions, Y))


similarity.register(
    "brain_language_nlp/ridge",
    partial(wrap_ridge_with_r2, ridge_tools.ridge),
)

# similarity.register(
#     "brain_language_nlp/ridge_sk",
#     partial(wrap_ridge_with_r2, ridge_tools.ridge_sk),
# )

# similarity.register(
#     "brain_language_nlp/ridge_svd",
#     partial(wrap_ridge_with_r2, ridge_tools.ridge_svd),
# )

similarity.register(
    "brain_language_nlp/kernel_ridge",
    partial(wrap_ridge_with_r2, ridge_tools.kernel_ridge),
    )

# similarity.register(
#     "brain_language_nlp/kernel_ridge_svd",
#     partial(wrap_ridge_with_r2, ridge_tools.kernel_ridge_svd),
# )

# similarity.register(
#     "measure/brain_language_nlp/cross_val_ridge",
#     # TODO: return the weights, still need to eval
#     ridge_tools.cross_val_ridge,
# )