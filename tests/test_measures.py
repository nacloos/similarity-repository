import pytest
import numpy as np

import similarity

seeds = np.arange(10)

names = similarity.make("measure", variants=True, return_config=True).keys()
measures = [similarity.make(f"measure.{name}") for name in names]
test_inputs = [
    (measure, seed) for measure in measures for seed in seeds
]
test_input_vars = ["measure", "seed"]


def generate_data():
    X = np.random.randn(100, 30)
    Y = np.random.randn(100, 30)
    return X, Y


def generate_3d_data():
    X = np.random.randn(100, 5, 30)
    Y = np.random.randn(100, 5, 30)
    return X, Y


@pytest.mark.parametrize(["measure"], [(measure,) for measure in measures])
def test_measures(measure):
    X, Y = generate_data()
    score = measure.fit_score(X=X, Y=Y)
    assert isinstance(score, float)


@pytest.mark.parametrize(["measure"], [(measure,) for measure in measures])
def test_measures_3d_data(measure):
    X, Y = generate_3d_data()
    score = measure.fit_score(X=X, Y=Y)
    assert isinstance(score, float)


def test_backends():
    backend_by_measure = similarity.make(package="backend:backends", key="backend_by_measure")

    X, Y = generate_data()
    for measure_name, backends in backend_by_measure.items():
        for backend_name in backends:
            measure = similarity.make(f"backend.{backend_name}.measure.{measure_name}")
            assert isinstance(measure, similarity.Measure), f"Expected type Measure, got '{measure}'"
            score = measure.fit_score(X, Y)
            assert isinstance(score, float)


# TODO: test invariance properties automatically based on measure properties
# def test_distance():
#     def test_equivalence(measure):
#         X = np.random.randn(100, 10)
#         d = measure.fit_score(X=X, Y=X)
#         assert np.allclose(d, 0)

#     def test_symmetry(measure):
#         X = np.random.randn(100, 10)
#         Y = np.random.randn(100, 10)
#         d1 = measure.fit_score(X=X, Y=Y)
#         d2 = measure.fit_score(X=Y, Y=X)
#         assert np.allclose(d1, d2)

#     def test_triangle_inequality(measure):
#         X = np.random.randn(100, 10)
#         Y = np.random.randn(100, 10)
#         Z = np.random.randn(100, 10)
#         d1 = measure.fit_score(X=X, Y=Y)
#         d2 = measure.fit_score(X=X, Y=Z)
#         d3 = measure.fit_score(X=Y, Y=Z)
#         assert d1 <= d2 + d3

#     n_repetitions = 10
#     for name in names:
#         measure = similarity.make(f"{name}")
#         for i in range(n_repetitions):
#             test_equivalence(measure)
#             test_symmetry(measure)
#             test_triangle_inequality(measure)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_isotropic_scaling_invariance(measure, seed):
#     X = np.random.randn(100, 10)
#     Y = np.random.randn() * X
#     ref_score = measure.fit_score(X=X, Y=X)
#     tsf_score = measure.fit_score(X=X, Y=Y)
#     assert np.allclose(ref_score, tsf_score)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_invertible_linear_invariance(measure, seed):
#     X = np.random.randn(100, 10)
#     T = np.random.rand(10, 10)
#     # Ensuring the matrix is invertible
#     while np.linalg.det(T) == 0:
#         T = np.random.rand(10, 10)
#     Y = X @ T
#     ref_score = measure.fit_score(X=X, Y=X)
#     tsf_score = measure.fit_score(X=X, Y=Y)
#     assert np.allclose(ref_score, tsf_score)


# @pytest.mark.parametrize(test_input_vars, test_inputs)
# def test_orthogonal_invariance(measure, seed):
#     X = np.random.randn(100, 10)
#     Q = ortho_group.rvs(10, size=1)

#     assert np.allclose(np.dot(Q, Q.T), np.eye(10))

#     ref_score = measure.fit_score(X=X, Y=X)
#     tsf_score = measure.fit_score(X=X, Y=X @ Q)
#     assert np.allclose(ref_score, tsf_score)
