import numpy as np
from sklearn.decomposition import PCA


def transpose(X, Y):
    assert len(X.shape) == 2, "Expected 2 dimensions, found {}".format(len(X.shape))
    assert len(Y.shape) == 2, "Expected 2 dimensions, found {}".format(len(Y.shape))
    return X.T, Y.T


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
    # take the arcosine to get a proper distance metric
    return np.arccos(score)


def angular_dist_to_score(score):
    normalized_score = 1 - score/(np.pi/2)
    return normalized_score