import numpy as np

import similarity

def test_measures():
    # some measures raise error if second dim is less than 25
    X = np.random.rand(30, 25)
    Y = np.random.rand(30, 25)

    measures = similarity.make("measure/*/*")
    for measure_id, measure in measures.items():
        print(measure_id)
        score = measure(X, Y)
        print(f"score: {score}")


if __name__ == "__main__":
    test_measures()
