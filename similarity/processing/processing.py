import numpy as np
from sklearn.decomposition import PCA


def transpose(X, Y):
    assert len(X.shape) == 2, "Expected 2 dimensions, found {}".format(len(X.shape))
    assert len(Y.shape) == 2, "Expected 2 dimensions, found {}".format(len(Y.shape))
    return X.T, Y.T


def array_to_tensor(X, Y):
    import torch
    if not isinstance(X, torch.Tensor):
        X = torch.Tensor(X)
    if not isinstance(Y, torch.Tensor):
        Y = torch.Tensor(Y)
    return X, Y


def tensor_to_float(score):
    return score.detach().item()


def square_score(score):
    return score**2


def sqrt_score(score):
    return np.sqrt(score)


def cosine_score(score):
    return np.cos(score)


def arccos_score(score):
    return np.arccos(score)


def one_minus_score(score):
    return 1 - score


def flatten_3d_to_2d(X, Y):
    """
    reshape X with 3 dimensions (n_timesteps x n_trials x n_neurons) to 2 dimensions
    """
    def _flatten(X):
        assert isinstance(X, np.ndarray), "Expected np.ndarray, found {}".format(type(X))
        if len(X.shape) == 3:
            X = X.reshape(X.shape[0] * X.shape[1], X.shape[2])
            return X
        elif len(X.shape) == 2:
            return X
        else:
            raise ValueError("shape of X should be 2 or 3, but found {}".format(len(X.shape)))
    return _flatten(X), _flatten(Y)


def pca_preprocessing(X, Y, **kwargs):
    def _pca(X):
        pca = PCA(**kwargs)
        pca.fit(X)
        return pca.transform(X)
    
    return _pca(X), _pca(Y)


def angular_dist(score):
    # take the arcosine to get a proper distance measure
    return np.arccos(score)


def angular_dist_to_score(score):
    normalized_score = 1 - score/(np.pi/2)
    return normalized_score


def angular_to_euclidean_shape_measure(X, Y, score):
    """
    shape-measure-angular: arccos(<X, YQ>/(||X|| ||Y||)))
    shape-measure-euclidean: ||X - YQ||
    Ref: (Williams, 2021), (Lange, 2023)
    """
    X_norm = np.linalg.norm(X, ord="fro")
    Y_norm = np.linalg.norm(Y, ord="fro")
    return np.sqrt(X_norm**2 + Y_norm**2 - 2 * X_norm * Y_norm * np.cos(score))


def euclidean_to_angular_shape_measure(X, Y, score):
    X_norm = np.linalg.norm(X, ord="fro")
    Y_norm = np.linalg.norm(Y, ord="fro")
    return np.arccos((X_norm**2 + Y_norm**2 - score**2) / (2 * X_norm * Y_norm))


def angular_measure_to_normalize_scored(score):
    return 1 - score/(np.pi/2)


if __name__ == "__main__":
    from netrep.measures import Linearmeasure

    procrustes_angular = Linearmeasure(alpha=1, score_method="angular")
    procrustes_euclidean = Linearmeasure(alpha=1, score_method="euclidean")

    X, Y = np.random.randn(100, 10), np.random.randn(100, 10)

    score_angular = procrustes_angular.fit_score(X, Y)
    score_euclidean = procrustes_euclidean.fit_score(X, Y)    
    print(score_angular)
    print(score_euclidean)
    print(euclidean_to_angular_shape_measure(X, Y, score_euclidean))
    print(angular_to_euclidean_shape_measure(X, Y, score_angular))
