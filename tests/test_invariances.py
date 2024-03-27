"""
Automatically test invariance properties of measures.
Experimental - not yet ready to be included in automated tests.
"""
from pathlib import Path
import pytest
import numpy as np
import pandas as pd
from scipy.stats import ortho_group

import similarity


save_dir = Path(__file__).parent / ".." / "figures"


eps = 1e-5

def generate_X():
    return np.random.rand(10, 15, 50)


def test_distance():
    def test_equivalence(measure):
        X = np.random.randn(100, 10)
        d = measure(X=X, Y=X)
        assert np.allclose(d, 0)

    def test_symmetry(measure):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        d1 = measure(X=X, Y=Y)
        d2 = measure(X=Y, Y=X)
        assert np.allclose(d1, d2)

    def test_triangle_inequality(measure):
        X = np.random.randn(100, 10)
        Y = np.random.randn(100, 10)
        Z = np.random.randn(100, 10)
        d1 = measure(X=X, Y=Y)
        d2 = measure(X=X, Y=Z)
        d3 = measure(X=Y, Y=Z)
        assert d1 <= d2 + d3

    n_repetitions = 10
    for name in similarity.make("measure.*.*").keys():
        measure = similarity.make(f"{name}")
        for i in range(n_repetitions):
            test_equivalence(measure)
            test_symmetry(measure)
            test_triangle_inequality(measure)


def test_isotropic_scaling_invariance(measure, seed):
    X = generate_X()
    Y = np.random.randn() * X

    # add a very small amount of noise because get some NaNs when taking arccos of something too close to 1
    Y += eps * np.random.randn(*Y.shape)

    ref_score = measure(X=X, Y=X)
    tsf_score = measure(X=X, Y=Y)
    return ref_score, tsf_score


def test_invertible_linear_invariance(measure, seed):
    X = generate_X()
    T = np.random.rand(X.shape[-1], X.shape[-1])
    # ensure matrix is invertible
    while np.linalg.det(T) == 0:
        T = np.random.rand(X.shape[-1], X.shape[-1])
    Y = X @ T

    # add a very small amount of noise because get some NaNs when taking arccos of something too close to 1
    Y += eps * np.random.randn(*Y.shape)

    ref_score = measure(X=X, Y=X)
    tsf_score = measure(X=X, Y=Y)
    return ref_score, tsf_score


def test_orthogonal_invariance(measure, seed):
    X = generate_X()
    Q = ortho_group.rvs(X.shape[-1], size=1)

    assert np.allclose(np.dot(Q, Q.T), np.eye(X.shape[-1]))

    Y = X @ Q
    # add a very small amount of noise because get some NaNs when taking arccos of something too close to 1
    Y += eps * np.random.randn(*Y.shape)

    ref_score = measure(X=X, Y=X)
    tsf_score = measure(X=X, Y=Y)
    return ref_score, tsf_score


def test_translation_invariance(measure, seed):
    X = generate_X()
    # translation along last axis
    if len(X.shape) == 3:
        Y = X + np.random.rand(1, 1, X.shape[-1])
    else:
        Y = X + np.random.rand(1, X.shape[-1])

    # add a very small amount of noise because get some NaNs when taking arccos of something too close to 1
    Y += eps * np.random.randn(*Y.shape)

    ref_score = measure(X=X, Y=X)
    tsf_score = measure(X=X, Y=Y)
    return ref_score, tsf_score


def test_permutation_invariance(measure, seed):
    X = generate_X()
    idx = np.random.permutation(X.shape[-1])
    if len(X.shape) == 3:
        Y = X[:, :, idx]
    else:
        Y = X[:, idx]

    # add a very small amount of noise because get some NaNs when taking arccos of something too close to 1
    Y += eps * np.random.randn(*Y.shape)

    ref_score = measure(X=X, Y=X)
    tsf_score = measure(X=X, Y=Y)
    return ref_score, tsf_score


invariance_tests = {
    "isotropic-scaling": test_isotropic_scaling_invariance,
    "invertible-linear": test_invertible_linear_invariance,
    "orthogonal": test_orthogonal_invariance,
    "translation": test_translation_invariance,
    "permutation": test_permutation_invariance,
}

if __name__ == "__main__":
    measures = similarity.make("measure.*.*")

    # TODO: why so many NaNs?
    results = {}
    for measure_id, measure in measures.items():
        backend = measure_id.split(".")[1]
        measure_name = measure_id.split(".")[-1]
        card = similarity.make(f"card.{measure_name}")
        print(measure_id, card)

        res = {}
        for invariance in invariance_tests.keys():
            try:
                ref_score, score = invariance_tests[invariance](measure, 0)
                res[invariance] = np.mean(np.abs(ref_score - score) / np.abs(ref_score))
            except Exception as e:
                print(f"Error in {measure_id} for {invariance}: {e}")
                res[invariance] = np.NaN
            print(res[invariance])
        results[measure_id.replace("measure.", "")] = res

    results_df = pd.DataFrame(results)
    # order columns alphabetically
    results_df = results_df.reindex(sorted(results_df.columns), axis=1)
    results_df.to_csv(save_dir / "invariance_results.csv")
    print(results_df)

    # plot measure vs invariance heatmap
    import matplotlib.pyplot as plt
    import seaborn as sns

    plt.figure(figsize=(30, 20))
    sns.heatmap(results_df, cmap="viridis", vmin=0, linewidths=1, cbar=True, linecolor="white", cbar_kws={"shrink": 0.05})
    plt.ylabel("Invariances")
    plt.xlabel("Measures")
    plt.gca().xaxis.tick_top()
    plt.gca().xaxis.set_label_position('top')
    plt.xticks(rotation=45, ha='left')
    plt.yticks(rotation=0, va='center')
    plt.axis('scaled')
    plt.tight_layout()
    plt.savefig(save_dir / "invariance_results.png", transparent=False, bbox_inches='tight', dpi=300)



