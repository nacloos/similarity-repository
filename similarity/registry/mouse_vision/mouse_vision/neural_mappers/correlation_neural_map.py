import numpy as np
import sklearn.linear_model as lm

from . import neural_map_base as nm

__all__ = ["CorrelationNeuralMap"]

class CorrelationNeuralMap(nm.NeuralMapBase):
    """
    This is the implementation for performing a correlation based fitting procedure
    between source and target. Here we select the neuron that best correlates with
    each individual target.
    """
    def __init__(self, percentile=100, identity=True,
                 regression_type="Ridge", regression_kwargs={}):
        super(CorrelationNeuralMap, self).__init__()
        # Size of subset of source features that correlate well with targets.
        # Default is 100th percentile, meaning only select the best source
        # feature.
        self._regression_type = regression_type
        self._regression_kwargs = regression_kwargs
        self._percentile = percentile
        self._identity = identity
        if identity:
            assert percentile == 100, f"Cannot use identity mapping if percentile != 100."

        self._corrs = None
        self._mappers = None

    def _gather_correlations(self, X, Y):
        """
        Subroutine to gather correlations between source and targets.

        Updates:
            self._corrs : (numpy.ndarray) of dimensions (source, targets). Entry ij
                          is the correlation between the column i in the source
                          features and column j in the targets.
        """
        assert X.ndim == Y.ndim == 2
        assert X.shape[0] == Y.shape[0]
        assert self._n_targets is not None

        # First, remove column means
        X_ = X - X.mean(axis=0)
        Y_ = Y - Y.mean(axis=0)

        # Second, normalize each column vector to norm 1
        X_ = X_ / np.linalg.norm(X_, axis=0)
        Y_ = Y_ / np.linalg.norm(Y_, axis=0)

        # Now compute dot product = Pearson correlation
        self._corrs = np.dot(X_.T, Y_) # (source, target)

    def fit(self, X, Y):
        """
        Performs the correlation based fitting procedure.

        Inputs:
            X : (numpy.ndarray) of dimensions (num_samples, num_features)
            Y : (numpy.ndarray) of dimensions (num_samples, num_targets)
        """
        assert X.shape[0] == Y.shape[0], \
            f"Source and target sample dimension do not match: {X.shape} and {Y.shape}"

        # If target is a one-dimensional vector, make it into a column vector
        if Y.ndim == 1:
            Y = Y[:,np.newaxis]

        assert X.ndim == 2
        assert Y.ndim == 2
        self._n_source = X.shape[1]
        self._n_targets = Y.shape[1]

        self._gather_correlations(X, Y)
        assert self._corrs.shape == (self._n_source, self._n_targets)

        # For each target neuron, find the source feature that maximally
        # correlates with it.
        self._mappers = list()
        for i in range(self._n_targets):

            source_idxs = self._get_sources(i)
            assert source_idxs.shape[0] == self._n_source
            source_features = X[:,source_idxs]

            # If identity mapping, then we only want one source neuron
            if self._identity:
                assert source_idxs.sum() >= 1 and source_idxs.ndim == 1
                best_source_idx = np.argwhere(source_idxs == True).flatten()
                assert best_source_idx.size >= 1
                mapper = best_source_idx[0] # We append the best source idx, taking the first one if there are ties
            else:
                if source_features.ndim == 1:
                    source_features = source_features[:,np.newaxis]

                mapper = getattr(lm, self._regression_type)(**self._regression_kwargs)
                mapper.fit(source_features, Y[:,i][:,np.newaxis])

            self._mappers.append(mapper)

        self._fitted = True

    def _get_sources(self, target_idx):
        # Returns array of booleans of length number of sources
        assert self._corrs.shape == (self._n_source, self._n_targets)
        corrs = self._corrs[:,target_idx]
        percentile = np.nanpercentile(corrs, self._percentile)
        source_idxs = (corrs >= percentile)
        if (~source_idxs).all():
            # In this case, the source corrs were all nan, so nanpercentile returned all False.
            # This was because on some trials the target neuron can be constant (0) firing rate,
            # so no source unit is better than any other in that case to subselect from.
            # Thus, we select the last 1-percentile in terms of rank order of source units.
            corrs = np.arange(len(corrs))
            percentile = np.nanpercentile(corrs, self._percentile)
            source_idxs = (corrs >= percentile)
        return source_idxs

    def predict(self, X):
        assert self._n_source is not None, "Fit procedure was not run."
        assert self._n_targets is not None, "Fit procedure was not run."
        assert self._mappers is not None, "Fit procedure was not run."
        assert self._fitted == True, "Fitting failed."
        assert len(self._mappers) == self._n_targets
        assert X.shape[1] == self._n_source, \
            f"Number of train features ({self._n_source}) does not align with number of test features ({X.shape[1]})"

        n_samples = X.shape[0]
        preds = np.empty((n_samples, self._n_targets)) + np.NaN
        for i, mapper in enumerate(self._mappers):
            if self._identity:
                assert isinstance(mapper, np.int64) # mapper contains the source index
                curr_pred = X[:,mapper]
            else:
                source_idxs = self._get_sources(i) # TODO: Cache source idxs
                assert source_idxs.shape[0] == X.shape[1]
                source_features = X[:,source_idxs]
                assert mapper.coef_.shape[-1] == source_features.shape[1], \
                    f"Linear map takes in {mapper.coef_.shape[-1]} features, not {source_features.shape[1]}."
                curr_pred = mapper.predict(source_features).flatten()

                assert curr_pred.ndim == 1 and curr_pred.shape[0] == n_samples
            preds[:,i] = curr_pred

        return preds


