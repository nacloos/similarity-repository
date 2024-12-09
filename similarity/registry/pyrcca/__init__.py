# https://github.com/gallantlab/pyrcca
import numpy as np
from .rcca.rcca import CCA

import similarity


# TODO: register other variants
# TODO: eigh in rcca.py raises error
def score(X, Y):
    cca = CCA()
    X = X - X.mean(0)
    Y = Y - Y.mean(0)
    cca.train([X, Y])
    scores = cca.validate([X, Y])
    return np.mean([np.mean(s) for s in scores])

similarity.register("pyrcca/cca", score)
