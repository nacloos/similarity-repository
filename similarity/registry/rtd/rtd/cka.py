import numpy as np
from math import sqrt

def hsic(P, Q):
    PPt = P @ np.transpose(P)
    QQt = Q @ np.transpose(Q)
    n = P.shape[0]
    E = np.eye(n) - np.ones((n, n)) / n
    
    hsic = np.trace(PPt @ E @ QQt @ E) / (n - 1) ** 2
    
    return hsic

def cka(P, Q):
    pp = sqrt(hsic(P, P) + 1e-10)
    qq = sqrt(hsic(Q, Q) + 1e-10)
    return hsic(P, Q) / pp / qq
