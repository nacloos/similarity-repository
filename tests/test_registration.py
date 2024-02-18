import numpy as np

from similarity import make


def test_measures():
    measures = make("measure.sim_metric.*")
    for measure_id, measure in measures.items():
        print(measure)
        X = np.random.randn(100, 30)
        Y = np.random.randn(100, 30)

        score = measure(X, Y)
        print(f"measure_id: {measure_id}, score: {score}")
        assert isinstance(score, float)

    measures = make("measure.netrep.*")
    for measure_id, measure in measures.items():
        X = np.random.randn(100, 30)
        Y = np.random.randn(100, 30)

        score = measure(X, Y)
        print(f"measure_id: {measure_id}, score: {score}")
        assert isinstance(score, float)


test_measures()
