import numpy as np
from . import neural_map_base as nm
from . utils import map_from_str
from collections import OrderedDict

__all__ = ["PipelineNeuralMap"]

class PipelineNeuralMap(nm.NeuralMapBase):
    def __init__(self,
                 map_type=None, map_kwargs={}):
        assert map_type is not None
        self._map_type = map_type
        self._map_kwargs = map_kwargs
        self._map_func = map_from_str(map_type)
        self.map = None

    def score(self, Y, Y_pred,
              spec_score_agg_func=np.mean):
        if isinstance(Y, dict):
            assert isinstance(Y_pred, dict) is True
            score_dict = OrderedDict()
            for spec_name in Y.keys():
                score_dict[spec_name] = self.map[spec_name].score(Y=Y[spec_name],
                                                                  Y_pred=Y_pred[spec_name])
            if spec_score_agg_func is None:
                return score_dict
            else:
                score_dict_agg = [np.array(s) for s in score_dict.values()]
                if score_dict_agg[0].ndim == 0:
                    score_arr = np.array(score_dict_agg)
                else:
                    score_arr = np.concatenate(score_dict_agg, axis=-1)
                return spec_score_agg_func(score_arr)
        else:
            return self.map.score(Y, Y_pred)

    def fit(self, X, Y):
        assert not isinstance(X, dict)
        if isinstance(Y, dict):
            self.map = OrderedDict()
            self._fitted = True
            for spec_name, spec_resp in Y.items():
                self.map[spec_name] = self._map_func(**self._map_kwargs)
                self.map[spec_name].fit(X, spec_resp)
                if not (self.map[spec_name]._fitted is True):
                    self._fitted = False # set to False if any map fails for a given animal
        else:
            self.map = self._map_func(**self._map_kwargs)
            self.map.fit(X, Y)
            self._fitted = self.map._fitted

    def predict(self, X):
        assert self._fitted is True, "Fitting failed."
        if isinstance(self.map, dict):
            predictions = OrderedDict()
            for spec_name, map_spec in self.map.items():
                predictions[spec_name] = map_spec.predict(X)
            return predictions
        else:
            return self.map.predict(X)


