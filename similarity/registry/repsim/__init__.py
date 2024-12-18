from functools import partial
import numpy as np
import torch

import repsim
from repsim.kernels import Linear, Laplace, SquaredExponential

import similarity

# TODO: sigma
# similarity.register("kernel/repsim/linear", Linear())
# similarity.register("kernel/repsim/laplace", Laplace())
# similarity.register("kernel/repsim/rbf", SquaredExponential())


def cka_measure(X: np.ndarray, Y: np.ndarray, sigma: float = 1.0, unbiased: bool = False, kernel: str = "rbf"):
    if kernel == "linear":
        kernel = Linear()
    elif kernel == "rbf":
        # convert sigma to length scale
        kernel = SquaredExponential(length_scale=sigma * np.sqrt(2))
    elif kernel == "laplace":
        # convert sigma to length scale
        kernel = Laplace(length_scale=sigma * np.sqrt(2))
    else:
        raise ValueError(f"Unknown kernel {kernel}")
    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    score = repsim.compare(x=X, y=Y, method="angular_cka", kernel=kernel, use_unbiased_hsic=unbiased)
    return float(score)


similarity.register("repsim/AngularCKA.linear", partial(cka_measure, kernel="linear", unbiased=False))
similarity.register("repsim/AngularCKA.unb.linear", partial(cka_measure, kernel="linear", unbiased=True))
similarity.register("repsim/AngularCKA.SqExp[{sigma}]", partial(cka_measure, kernel="rbf", unbiased=False))
similarity.register("repsim/AngularCKA.unb.SqExp[{sigma}]", partial(cka_measure, kernel="rbf", unbiased=True))
similarity.register("repsim/AngularCKA.Laplace[{sigma}]", partial(cka_measure, kernel="laplace", unbiased=False))
similarity.register("repsim/AngularCKA.unb.Laplace[{sigma}]", partial(cka_measure, kernel="laplace", unbiased=True))


# shape metrics
def shape_measure(X: np.ndarray, Y: np.ndarray, name: str, alpha: float = 1.0) -> float:
    # no PCA
    p = X.shape[1]

    # assert p == Y.shape[1], "X and Y must have the same number of features"

    X = torch.as_tensor(X)
    Y = torch.as_tensor(Y)
    score = repsim.compare(x=X, y=Y, method=name, alpha=alpha, p=p)
    return float(score)


similarity.register("repsim/ShapeMetric[{alpha}][angular]", partial(shape_measure, name="angular_shape_metric"))
similarity.register("repsim/ShapeMetric[{alpha}][euclidean]", partial(shape_measure, name="euclidean_shape_metric"))

