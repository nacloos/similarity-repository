


class Metric:
    def __init__(self, metric, fit_score):
        self._metric = metric
        self._fit_score = fit_score

    def fit(self, X, Y):
        # TODO
        pass

    def score(self, X, Y):
        #TODO
        pass

    def fit_score(self, X, Y):
        out = self._fit_score(metric=self._metric, X=X, Y=Y)
        # TODO: specify that in config
        return out["score"]
