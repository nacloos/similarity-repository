import numpy as np

from sklearn.cross_decomposition import PLSRegression
from sklearn.model_selection import KFold

from . import neural_map_base as nm

__all__ = ["PLSNeuralMap"]

class PLSNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing PLS regression between source and target.
    """
    def __init__(self, n_components=None, scale=False, fit_per_target_unit=False, verbose=False):
        super(PLSNeuralMap, self).__init__()
        self._n_components = n_components
        self._scale = scale
        self._pls = None
        self._fit_per_target_unit = fit_per_target_unit
        self._verbose = verbose

    def fit(self, X, Y):
        # If target is a one-dimensional vector, make it into a column vector
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        assert X.shape[0] == Y.shape[0], \
            f"Features and targets sample dimension does not match: {X.shape} and {Y.shape}"
        assert X.ndim == Y.ndim == 2

        self._n_source = X.shape[1]
        self._n_targets = Y.shape[1]

        fit_kwargs = {"n_components": self._n_components, "scale": self._scale}
        assert self._n_components is not None, "Need to specify number of PLS components."
        if self._fit_per_target_unit:
            # fit a separate PLS mapping per target neuron
            self._pls = list()
            for i in range(self._n_targets):
                self._pls.append(self._fit(X, Y[:, i], fit_kwargs))

            # if pls failed for every target neuron then we say it failed altogether
            if all(curr_pls is None for curr_pls in self._pls):
                self._pls = None
            else:
                self._fitted = True
        else:
            self._pls = self._fit(X, Y, fit_kwargs)
            if self._pls is not None:
                self._fitted = True

    def _fit(self, X, Y, fit_kwargs):
        """
        Performs the PLS regression fitting.

        Inputs:
            X : (numpy.ndarray) of dimensions (num_stimuli, num_features)
            Y : (numpy.ndarray) of dimensions (num_stimuli, num_targets)
        """
        assert Y.ndim == 2

        try:
            pls = PLSRegression(**fit_kwargs)
            pls.fit(X, Y)
            return pls
        except: # If SVD does not converge
            return None

    def predict(self, X):
        """
        Performs response prediction with features from test set.

        Inputs:
            X : (numpy.ndarray) of dimensions (num_stimuli, num_features)

        Returns:
            preds : (numpy.ndarray) of dimensions (num_stimuli, num_targets)
        """
        assert self._n_source is not None, "Fit procedure was not run."
        assert self._n_targets is not None, "Fit procedure was not run."
        assert X.shape[1] == self._n_source, \
            f"Number of train features ({self._n_source}) does not align with number of test features ({X.shape[1]})"

        n_samples = X.shape[0]
        if self._verbose:
            print(f"Number of components in PLS: {self._pls.n_components}")

        if self._fitted:
            assert self._pls is not None
            if self._fit_per_target_unit:
                assert(isinstance(self._pls, list) is True)
                assert(len(self._pls) == self._n_targets)
                preds = np.empty((n_samples, self._n_targets)) + np.NaN
                for i, curr_pls in enumerate(self._pls):
                    if curr_pls is not None:
                        curr_pred = curr_pls.predict(X).flatten()
                        assert curr_pred.ndim == 1 and curr_pred.shape[0] == n_samples
                        preds[:, i] = curr_pred
            else:
                preds = self._pls.predict(X)
        else:
            assert self._pls is None
            if self._verbose:
                print(f"[WARNING] PLS regression fitting failed.")
            preds = np.zeros((n_samples, self._n_targets)) + np.NaN

        return preds


