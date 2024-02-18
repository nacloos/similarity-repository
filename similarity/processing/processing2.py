import numpy as np
from sklearn.decomposition import PCA

from similarity import register


@register("preprocessing.transpose")
def transpose(X):
    assert len(X.shape) == 2, "Expected 2 dimensions, found {}".format(len(X.shape))
    return X.T


@register("preprocessing.array_to_tensor")
def array_to_tensor(X):
    import torch
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    return X


@register("preprocessing.reshape2d")
def reshape2d(X):
    if len(X.shape) == 3:
        X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
    elif len(X.shape) == 2:
        pass
    else:
        raise ValueError("shape of X should be 2 or 3, but found {}".format(len(X.shape)))
    return X


@register("preprocessing.pca-var99")
def pca_var99(X):
    pca = PCA(n_components=0.99)
    return pca.fit_transform(X)


@register("preprocessing.pca-var95")
def pca_var95(X):
    pca = PCA(n_components=0.95)
    return pca.fit_transform(X)


@register("postprocessing.tensor_to_float")
def tensor_to_float(score):
    return score.detach().item()

@register("postprocessing.square")
def square_score(score):
    return score**2


@register("postprocessing.sqrt")
def sqrt_score(score):
    return np.sqrt(score)


@register("postprocessing.cos")
def cosine_score(score):
    return np.cos(score)


@register("postprocessing.arccos")
def arccos_score(score):
    return np.arccos(score)


@register("postprocessing.one_minus")
def one_minus_score(score):
    return 1 - score


@register("postprocessing.normalize_pi_half")
def angular_dist_to_score(score):
    normalized_score = score/(np.pi/2)
    return normalized_score


def angular_to_euclidean_shape_metric(X, Y, score):
    """
    shape-metric-angular: arccos(<X, YQ>/(||X|| ||Y||)))
    shape-metric-euclidean: ||X - YQ||
    Ref: (Williams, 2021), (Lange, 2023)
    """
    X_norm = np.linalg.norm(X, ord="fro")
    Y_norm = np.linalg.norm(Y, ord="fro")
    return np.sqrt(X_norm**2 + Y_norm**2 - 2 * X_norm * Y_norm * np.cos(score))


def euclidean_to_angular_shape_metric(X, Y, score):
    X_norm = np.linalg.norm(X, ord="fro")
    Y_norm = np.linalg.norm(Y, ord="fro")
    return np.arccos((X_norm**2 + Y_norm**2 - score**2) / (2 * X_norm * Y_norm))

