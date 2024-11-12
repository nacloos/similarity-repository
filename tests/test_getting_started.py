"""
Test the code examples in the README. 
Make sure to update the tests when changing the README.
"""
import numpy as np
import similarity


# generate some random data
X, Y = np.random.randn(100, 30), np.random.randn(100, 30)


def test_simple_make():
    # make a measure object
    measure = similarity.make("measure.netrep.procrustes-angular")
    score = measure(X, Y)


def test_measure_subgroups():
    measures = similarity.make("measure.netrep.*")
    for name, measure in measures.items():
        score = measure(X, Y)
        print(f"{name}: {score}")

    measures = similarity.make("measure.*.procrustes-angular")
    for name, measure in measures.items():
        score = measure(X, Y)
        print(f"{name}: {score}")


def test_register():
    @similarity.register("measure.my_package.my_measure", function=True)
    def my_measure(x, y):
        return x.reshape(-1) @ y.reshape(-1) / (np.linalg.norm(x) * np.linalg.norm(y))

    measure = similarity.make("measure.my_package.my_measure")
    score = measure(X, Y)


    @similarity.register("measure.my_package.my_measure2")
    class MyMeasure:
        def fit(self, X, Y):
            self.X_norm = np.linalg.norm(X)
            self.Y_norm = np.linalg.norm(Y)

        def score(self, X, Y):
            return X.reshape(-1) @ Y.reshape(-1) / (self.X_norm * self.Y_norm)

        def fit_score(self, x, y):
            self.fit(x, y)
            return self.score(x, y)

        def __call__(self, x, y):
            return self.fit_score(x, y)

    measure2 = similarity.make("measure.my_package.my_measure2")

    X_fit, Y_fit = np.random.randn(100, 30), np.random.randn(100, 30)
    X_val, Y_val = np.random.randn(100, 30), np.random.randn(100, 30)

    measure2.fit(X_fit, Y_fit)
    score = measure2.score(X_val, Y_val)


if __name__ == "__main__":
    test_simple_make()
    test_measure_subgroups()
    test_register()
    print("All tests passed!")
