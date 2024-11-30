# https://github.com/neuroailab/mouse-vision
import sys
from pathlib import Path
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))

from mouse_vision.reliability.metrics import rsa
from mouse_vision.neural_mappers import PLSNeuralMap, CorrelationNeuralMap

import similarity


similarity.register(
    "mouse_vision/rsa",
    rsa
)


def score(cls, **kwargs):
    def _score(X, Y):
        pls = cls(**kwargs)
        pls.fit(X, Y)
        Y_pred = pls.predict(X)
        scores = pls.score(Y, Y_pred)
        return scores.mean()
    return _score

similarity.register("mouse_vision/PLSNeuralMap", score(PLSNeuralMap, n_components=25))
similarity.register("mouse_vision/CorrelationNeuralMap", score(CorrelationNeuralMap))
