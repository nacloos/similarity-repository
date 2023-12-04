"""
Test the code examples in the README. 
Make sure to update the tests when changing the README.
"""
import numpy as np
import similarity


# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)


def test_simple_make():
    # make a particular measure
    measure = similarity.make("measure.procrustes")
    score = measure.fit_score(X, Y)
    print(score)


def test_measure_subgroups():
    # make all the measures
    measures = similarity.make("measure")
    for name, measure in measures.items():
        # all the measures have the same interface
        score = measure.fit_score(X, Y)
        print(f"{name}: {score}")


    # return_config=True returns the config instead of the instantiated object
    measure_configs = similarity.make("measure", return_config=True)
    # select desired subset
    score_ids = [k for k, cfg in measure_configs.items() if "score" in cfg["properties"]]
    # make the measures
    score_measures = {k: similarity.make(f"measure.{k}") for k in score_ids}

    for name, measure in score_measures.items():
        print(f"Score {name}: {measure.fit_score(X, Y)}")


    metric_ids = [k for k, cfg in measure_configs.items() if "metric" in cfg["properties"]]
    # make the measures
    metric_measures = {k: similarity.make(f"measure.{k}") for k in metric_ids}

    for name, measure in metric_measures.items():
        print(f"Metric {name}: {measure.fit_score(X, Y)}")


def test_choose_backend():
    # example of backend and measure
    backend_id = "repsim"
    measure_id = "procrustes"
    measure = similarity.make(f"backend.{backend_id}.measure.{measure_id}")
    score = measure.fit_score(X, Y)
    print(score)


def test_customize_interface():
    measure = similarity.make(
        "measure.procrustes",
        interface={
            # replaces the method fit_score with a __call__ method 
            "fit_score": "__call__"
        }
    )
    score = measure(X, Y)
    print(score)


def test_register():
    def my_metric(x, y):
        return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))

    # register the function with a unique id
    similarity.register(my_metric, "measure.my_metric.fit_score")

    metric = similarity.make("measure.my_metric")
    score = metric.fit_score(X, Y)
    print(score)

    class MyMetric:
        def fit(self, X, Y):
            pass

        def score(self, X, Y):
            return X.reshape(-1) @ Y.reshape(-1) / (np.linalg.norm(X) * np.linalg.norm(Y))

        def fit_score(self, x, y):
            self.fit(x, y)
            return self.score(x, y)

    similarity.register(MyMetric, "measure.my_metric2")

    metric2 = similarity.make("measure.my_metric2")
    metric2.fit(X, Y)
    metric2.score(X, Y)
    score = metric2.fit_score(X, Y)
    print(score)

# test_simple_make()
# test_measure_subgroups()
# test_choose_backend()
# test_customize_interface()
# test_register()
