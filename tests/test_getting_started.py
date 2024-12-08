"""
Test the code examples in the README. 
Make sure to update the tests when changing the README.
"""
import numpy as np
import similarity


# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)


def test_simple_make():
    import numpy as np
    import similarity

    # generate two datasets
    X, Y = np.random.randn(100, 30), np.random.randn(100, 30)

    # measure their similarity
    measure = similarity.make("measure/netrep/procrustes-distance=angular")
    score = measure(X, Y)


def test_measure_subgroups():
    measures = similarity.make("measure/netrep/*")
    for name, measure in measures.items():
        score = measure(X, Y)
        print(f"{name}: {score}")

    measures = similarity.make("measure/*/procrustes-distance=angular")
    for name, measure in measures.items():
        score = measure(X, Y)
        print(f"{name}: {score}")


def test_register():
    # register the function with a unique id
    def my_measure(x, y):
        return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))

    similarity.register("my_repo/my_measure", my_measure)

    # use it like any other measure
    measure = similarity.make("my_repo/my_measure")
    score = measure(X, Y)


if __name__ == "__main__":
    test_simple_make()
    test_measure_subgroups()
    test_register()
    print("All tests passed!")
