

class Metric:
    def __init__(self, metric, fit_score, score=None, fit=None):
        self._metric = metric
        self._score = score
        self._fit = fit
        self._fit_score = fit_score

    def fit(self, X, Y):
        if self._fit is not None:
            self._fit(metric=self._metric, X=X, Y=Y)

    def score(self, X, Y):
        return self._score(metric=self._metric, X=X, Y=Y)

    def fit_score(self, X, Y):
        if self._fit_score is not None:
            return self._fit_score(metric=self._metric, X=X, Y=Y)
        else:
            self.fit(X, Y)
            return self.score(X, Y)
