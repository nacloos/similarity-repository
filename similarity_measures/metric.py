


class Metric:
    def __init__(self, metric, fit_score):
        print("Metric: ", metric)
        print("Fit score: ", fit_score)
        self._metric = metric
        self._fit_score = fit_score

    def fit_score(self, *args, **kwargs):
        out = self._fit_score(metric=self._metric, *args, **kwargs)
        # TODO: specify that in config
        return out["score"]
