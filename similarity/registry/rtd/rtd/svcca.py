import numpy as np
from . import cca_core

def get_sv(acts1):
    cb1 = acts1 - np.mean(acts1, axis=0, keepdims=True)

    # Perform SVD
    Ub1, sb1, Vb1 = np.linalg.svd(cb1, full_matrices=False)

    d = get_threshold(sb1)
    svb1 = np.dot(sb1[:d]*np.eye(d), Vb1[:d])
    
    return svb1

def get_threshold(sb1):
    for d in range(sb1.shape[0]):
        if np.sum(sb1[0:d]) > 0.99 * np.sum(sb1):
            return d
        
    return sb1.shape[0]

def svcca(acts1, acts2):
    svb1 = get_sv(acts1)
    svb2 = get_sv(acts2)
    
    svcca_baseline = cca_core.get_cca_similarity(svb1, svb2, epsilon=1e-10, verbose=False)
    return (np.mean(svcca_baseline["cca_coef1"])) 
