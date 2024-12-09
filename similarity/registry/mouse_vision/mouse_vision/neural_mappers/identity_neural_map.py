from . import neural_map_base as nm

__all__ = ["IdentityNeuralMap"]

class IdentityNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing the identity map from source units.
    This mapper does not make sense when trying to map source to target units. It
    only makes sense for source to source.
    """
    def __init__(self):
        # Identity map only works for RSA metric. It doesn't make sense to use
        # the correlation score function since number of sources could be different
        # from the number of targets.
        super(IdentityNeuralMap, self).__init__(score_func="rsa")

    def fit(self, X, Y):
        # Don't need to do any fitting
        self._fitted = True
        self._n_source = X.shape[1]
        self._n_targets = Y.shape[1]

    def predict(self, X):
        # Return identity
        assert X.ndim == 2
        assert self._n_source == X.shape[1]
        return X

