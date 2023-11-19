import numpy as np
import similarity


names = ["procrustes", "cca", "svcca", "cka"]

for name in names:
    metric = similarity.make(f"{name}")
    print("Metric:", name)

    X = np.random.randn(100, 10)
    Y = np.random.randn(100, 10)
    print(metric.fit_score(X=X, Y=Y))

    X = np.random.randn(100, 5, 10)
    Y = np.random.randn(100, 5, 10)
    print(metric.fit_score(X=X, Y=Y))
    print()



