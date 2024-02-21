from collections import defaultdict
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

from similarity import Measure, register, make


@register("measure.cka-angular-score")
def _():
    def _fit_score(X, Y):
        measure = Measure("cka-angular")
        return 1 - measure(X, Y) / np.pi * 2
    return _fit_score

@register("measure.procrustes-angular-score")
def _():
    def _fit_score(X, Y):
        measure = Measure("procrustes-angular")
        return 1 - measure(X, Y) / np.pi * 2
    return _fit_score


#TODO: normal that procrustes and cka angular score are the same?
#TODO: 2d matrix like Chris did
#TODO: benchmark invariances (!= unittests because so measure are expected to fail)
#TODO: add pytorch measures and test equality with other measures
def benchmark_ind_gaussians():
    measure_ids = [
        "cka",
        "procrustes-angular-score",
        "cka-angular-score"
    ]

    # dim_values = np.arange(1, 5000, 10)
    dim_values = np.arange(1, 1000, 10)
    scores = defaultdict(list)
    for measure_id in measure_ids:
        measure = Measure(measure_id)

        for dim in dim_values:
            X = np.random.randn(100, dim)
            Y = np.random.randn(100, dim)

            score = measure(X, Y)
            scores[measure_id].append(score)

    plt.figure()
    for measure_id in measure_ids:
        plt.plot(dim_values, scores[measure_id], label=measure_id)
    plt.xlabel("Dimension")
    plt.ylabel("Score")
    plt.legend()
    plt.show()


@register("benchmark.metric-equality")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        Y = np.random.randn(*shape)
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=X, Y=Y)
        score += np.allclose(score1, score2)
    return score / n_repeats


@register("benchmark.metric-symmetry")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        Y = np.random.randn(*shape)
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=Y, Y=X)
        score += np.allclose(score1, score2)
    return score / n_repeats


@register("benchmark.metric-triangle_inequality")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        Y = np.random.randn(*shape)
        Z = np.random.randn(*shape)
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=X, Y=Z)
        score3 = measure(X=Y, Y=Z)
        score += score1 <= score2 + score3
    return score / n_repeats


@register("benchmark.invariance-isotropic_scaling")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        Y = X * np.random.randn()
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=X, Y=Y)
        score += np.allclose(score1, score2)
    return score / n_repeats


@register("benchmark.invariance-invertible_linear")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        T = np.random.rand(shape[1], shape[1])
        while np.linalg.det(T) == 0:
            T = np.random.rand(shape[1], shape[1])
        Y = X @ T
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=X, Y=Y)
        score += np.allclose(score1, score2)
    return score / n_repeats


@register("benchmark.invariance-orthogonal")
def _(measure, shape=(100, 30), n_repeats=10):
    score = 0
    for _ in range(n_repeats):
        X = np.random.randn(*shape)
        Q = np.random.rand(shape[1], shape[1])
        Q, _ = np.linalg.qr(Q)
        Y = X @ Q
        score1 = measure(X=X, Y=Y)
        score2 = measure(X=X, Y=Y)
        score += np.allclose(score1, score2)
    return score / n_repeats


if __name__ == "__main__":
    # benchmark_ind_gaussians()


    benchmark_ids = [
        "metric-equality",
        "metric-symmetry",
        "metric-triangle_inequality",
        "invariance-isotropic_scaling",
        "invariance-invertible_linear",
        "invariance-orthogonal"
    ]
    measure_ids = [
        "cca",
        "rsa",
        "procrustes-angular-score",
        "cka",
        "cka-angular-score",
    ]
    # TODO: cca, rsa not supposed to satisfy triangle inequality (not enough to test with random Gaussians?)
    # TODO: diff optimization to find test cases that fail?
    n_repeats = 10
    # shape = (100, 30)
    shape = (50, 40)
    results = defaultdict(list)
    for measure_id in measure_ids:
        measure = make(f"measure.{measure_id}")
        for benchmark_id in benchmark_ids:
            score = make(f"benchmark.{benchmark_id}", measure=measure, shape=shape, n_repeats=n_repeats)
            print(f"{measure_id} - {benchmark_id}: {score:.2f}")
            results[benchmark_id].append(score)

    import pandas as pd
    df = pd.DataFrame(results, index=measure_ids)
    print(df)
    # save to csv
    df.to_csv(Path(__file__).parent / "benchmark_results.csv", index=True)
