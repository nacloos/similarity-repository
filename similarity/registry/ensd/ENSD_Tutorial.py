# Code taken from https://github.com/camillerb/ENSD/blob/main/ENSD_Tutorial.ipynb
import numpy as np
from scipy.stats import ortho_group 
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D 
import pandas as pd
import random as rd
import copy
import tqdm
import math

## Define functions to compute ENSD and PR
def PR(X):
    return (np.trace(X.T@X)**2)/np.trace(X.T@X@X.T@X)
def TR(X,Y):
    return np.trace( ((X.T@X)/np.trace(X.T@X)) @ ((Y.T@Y)/np.trace(Y.T@Y)) )
def ENSD(X, Y):
    #input is data matrix
    return PR(X)*PR(Y)*TR(X,Y)#
def computeDist(X, Y):
    return (2/math.pi)*(np.arccos(ENSD(X,Y)/np.sqrt(PR(X)*PR(Y))))
def eigvecOverlap(X,Y):
    ux, sx, vx = np.linalg.svd(X, full_matrices=True)                       
    uy, sy, vy = np.linalg.svd(Y, full_matrices=True) 
    return np.square(ux.T@uy)
def gen_orthonormal(dim):
    H = np.random.rand(dim, dim)
    u, s, vh = np.linalg.svd(H, full_matrices=False)
    return u#ortho_group.rvs(dim)