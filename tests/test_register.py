import numpy as np
import similarity


def my_metric(x, y, param=None):
    if param is not None:
        return param
    return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))


def preprocess(X, Y):
    return X, Y


def test_register():
    similarity.register(my_metric, "measure.my_metric.fit_score")

    metric = similarity.make("measure.my_metric")
    X, Y = np.random.randn(100, 30), np.random.randn(100, 30)
    metric.fit_score(X, Y)


def test_preprocessing():
    similarity.register(my_metric, "measure.my_metric.fit_score")

    metric = similarity.make("measure.my_metric", preprocessing=[preprocess])
    print(metric._fit_score)
