import pytest
import numpy as np
from scipy.stats import ortho_group

import similarity


names = similarity.make("metric").keys()

metrics = [similarity.make(f"metric.{name}") for name in names]
seeds = np.arange(10)
test_inputs = [
    (metric, seed) for metric in metrics for seed in seeds
]
test_input_vars = ["metric", "seed"]


# TODO: test distance measures separately then score measures

def generate_data():
    X = np.random.randn(100, 30)
    Y = np.random.randn(100, 30)
    return X, Y


def generate_3d_data():
    X = np.random.randn(100, 5, 30)
    Y = np.random.randn(100, 5, 30)
    return X, Y

@pytest.mark.parametrize(["metric"], [(metric,) for metric in metrics])
def test_metrics(metric):
    X, Y = generate_data()
    score = metric.fit_score(X=X, Y=Y)
    assert isinstance(score, float)

    # for name in names:
    #     # metric = similarity.make(f"metric/{name}")
    #     metric = similarity.make(package="metric", key=name)

    #     X, Y = generate_data()
    #     score = metric.fit_score(X=X, Y=Y)
    #     assert isinstance(score, float)

    #     X = np.random.randn(100, 5, 30)
    #     Y = np.random.randn(100, 5, 30)
    #     score = metric.fit_score(X=X, Y=Y)
    #     assert isinstance(score, float)


@pytest.mark.parametrize(["metric"], [(metric,) for metric in metrics])
def test_metrics_3d_data(metric):
    X, Y = generate_3d_data()
    score = metric.fit_score(X=X, Y=Y)
    assert isinstance(score, float)


def test_backends():
    backend_by_metric = similarity.make(package="backend:backends", key="backend_by_metric")

    X, Y = generate_data()
    for metric_name, backends in backend_by_metric.items():
        for backend_name in backends:
            metric = similarity.make(f"backend.{backend_name}.metric.{metric_name}")
            assert isinstance(metric, similarity.Metric), f"Expected type Metric, got '{metric}'"
            score = metric.fit_score(X, Y)
            assert isinstance(score, float)

# TODO
# def test_distance():
#     def test_equivalence(metric):
#         X = np.random.randn(100, 10)
#         d = metric.fit_score(X=X, Y=X)
#         assert np.allclose(d, 0)

#     def test_symmetry(metric):
#         X = np.random.randn(100, 10)
#         Y = np.random.randn(100, 10)
#         d1 = metric.fit_score(X=X, Y=Y)
#         d2 = metric.fit_score(X=Y, Y=X)
#         assert np.allclose(d1, d2)

#     def test_triangle_inequality(metric):
#         X = np.random.randn(100, 10)
#         Y = np.random.randn(100, 10)
#         Z = np.random.randn(100, 10)
#         d1 = metric.fit_score(X=X, Y=Y)
#         d2 = metric.fit_score(X=X, Y=Z)
#         d3 = metric.fit_score(X=Y, Y=Z)
#         assert d1 <= d2 + d3

#     n_repetitions = 10
#     for name in names:
#         metric = similarity.make(f"{name}")
#         for i in range(n_repetitions):
#             test_equivalence(metric)
#             test_symmetry(metric)
#             test_triangle_inequality(metric)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_isotropic_scaling_invariance(metric, seed):
#     X = np.random.randn(100, 10)
#     Y = np.random.randn() * X
#     ref_score = metric.fit_score(X=X, Y=X)
#     tsf_score = metric.fit_score(X=X, Y=Y)
#     assert np.allclose(ref_score, tsf_score)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_invertible_linear_invariance(metric, seed):
#     X = np.random.randn(100, 10)
#     T = np.random.rand(10, 10)
#     # Ensuring the matrix is invertible
#     while np.linalg.det(T) == 0:
#         T = np.random.rand(10, 10)
#     Y = X @ T
#     ref_score = metric.fit_score(X=X, Y=X)
#     tsf_score = metric.fit_score(X=X, Y=Y)
#     assert np.allclose(ref_score, tsf_score)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_orthogonal_invariance(metric, seed):
#     X = np.random.randn(100, 10)
#     Q = ortho_group.rvs(10, size=1)

#     assert np.allclose(np.dot(Q, Q.T), np.eye(10))

#     ref_score = metric.fit_score(X=X, Y=X)
#     tsf_score = metric.fit_score(X=X, Y=X @ Q)
#     assert np.allclose(ref_score, tsf_score)
