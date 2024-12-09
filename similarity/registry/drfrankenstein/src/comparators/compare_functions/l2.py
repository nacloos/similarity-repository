import numpy as np


def l2(x1, x2):
    x1 = x1.flatten()
    x2 = x2.flatten()
    distance = np.mean((x1 - x2) ** 2)

    return distance
