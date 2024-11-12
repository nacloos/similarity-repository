import numpy as np

import similarity


def test_measures():
    """
    Make sure can create all measures and they run without error.
    """

    X = np.random.randn(15, 20, 30)
    Y = np.random.randn(15, 20, 30)

    measures = similarity.all_measures()
    print(len(measures))
    breakpoint()

    for measure_id in measures:
        print(f"Testing {measure_id}")
        # measure = similarity.make_measure(measure_id)
        measure = similarity.make(measure_id)

        score = measure(X, Y)
        print(f"score: {score}")
        print()


if __name__ == "__main__":
    test_measures()

