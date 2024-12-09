# Taken from https://github.com/anshksoni/NeuroAIMetrics/blob/main/utils/metrics.py
import numpy as np
from sklearn.decomposition import PCA

from scipy.stats import kendalltau
from torchmetrics.functional import pairwise_cosine_similarity
import torch
from sklearn.linear_model import RidgeCV
from sklearn.cross_decomposition import PLSRegression
import gc
from sklearn.model_selection import KFold
import ot
import scipy
from fastprogress.fastprogress import progress_bar

np.seterr(invalid='ignore')

# Tradeoff between RAM and computation time
def many_pairwise_correlation(A,B):
    memsize=10000 # change this value, higher means faster but more RAM
    corr=[]
    lens=[]
    for i in range(0, A.shape[1], memsize):
        model_corrs=pairwise_correlation(A[:,i:i+memsize],B[:,i:i+memsize])
        corr.append(np.nanmean(np.diag(model_corrs)))
        lens.append(A[:,i:i+memsize].shape[1])
    corr=np.sum(np.multiply(np.array(corr),np.array(lens)))/np.sum(lens)
    gc.collect()
    return corr
            
        
    

def pairwise_correlation(A, B):
        am = A - np.mean(A, axis=0, keepdims=True)
        bm = B - np.mean(B, axis=0, keepdims=True)
        return am.T @ bm /  (np.sqrt(
            np.sum(am**2, axis=0,
                keepdims=True)).T * np.sqrt(
            np.sum(bm**2, axis=0, keepdims=True)))
        
        
def normalize(X):
    
    mean = np.nanmean(X, 0) 

    stddev = np.nanstd(X, 0) 
    X_zm = X - mean    
    X_zm_unit = X_zm / stddev
    X_zm_unit[np.isnan(X_zm_unit)] = 0
    return X_zm_unit

def LinearPredictivity(X,Y):

    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        predictor = RidgeCV(alphas=np.logspace(-8,8,17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])

        y_pred = predictor.predict(normalize(X[test_idx]))

        corr=many_pairwise_correlation(y_pred,Y[test_idx])
        del predictor
        gc.collect()
        all_corrs.append(corr)

        
        
 
    return all_corrs
        
def reverseLinearPredictivity(Y,X):
        
    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        predictor = RidgeCV(alphas=np.logspace(-8,8,17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])
        # print('fit')
        y_pred = predictor.predict(normalize(X[test_idx]))
        
        corr=many_pairwise_correlation(y_pred,Y[test_idx])
        del predictor
        gc.collect()
        all_corrs.append(corr)
 
    return all_corrs
        
        
def SoftMatching(X,Y):
    rem=[]
    for i in range(X.shape[1]):
        if np.all(X[:,i]==0):
            rem.append(i)
    X=np.delete(X, rem, axis=1)
    rem=[]
    for i in range(Y.shape[1]):
        if np.all(Y[:,i]==0):
            rem.append(i)
    Y=np.delete(Y, rem, axis=1)
            
       
    x = torch.tensor(X)
    y = torch.tensor(Y)

    C = 1-pairwise_cosine_similarity(x.float().T - torch.nanmean(x.float().T,1)[:,None], y.float().T - torch.nanmean(y.float().T,1)[:,None])

    a = torch.ones(x.shape[1])/x.shape[1]
    b = torch.ones(y.shape[1])/y.shape[1]
    x = x.float()
    y = y.float()
    a = np.ones(x.shape[1])/x.shape[1]
    b = np.ones(y.shape[1])/y.shape[1]
    T = ot.emd(torch.tensor(a), torch.tensor(b), C)
    score = (1-np.nansum(T*C.numpy()))
    del C
    del x
    del y
    del a
    del b
    del T
    gc.collect()

    return [score]

def pairwisematching(X,Y):
    
    X=X.astype(np.float32)
    Y=Y.astype(np.float32)
    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        m_codes_ = [normalize(X[train_idx]),normalize(X[test_idx])] 
        gts_ = [Y[train_idx],Y[test_idx]]  
        
        m_codes, gts = [], []
        for code in m_codes_:
            m_codes.append(torch.tensor(code))
        for code in gts_:
            gts.append(torch.tensor(code))

        A = m_codes[0]

        B = gts[0]
        N = B.shape[0]

        sA = A.sum(0)
        sB = B.sum(0)

        p1 = N*torch.einsum('ij,ik->kj',A,B)
        p2 = sA*sB[:,None]
        p3 = N*((B**2).sum(0)) - (sB**2)
        p4 = N*((A**2).sum(0)) - (sA**2)

        pcorr = ((p1 - p2)/torch.sqrt(p4*p3[:,None]))

        pcorr[torch.isnan(pcorr)] = 0

        test_mat = m_codes[1]
        test_mat[torch.isinf(test_mat)] = 0
        indices = torch.argmax(pcorr,1)
        test_mat = test_mat.cpu().detach().numpy()
        gt = gts[1].cpu().detach().numpy()

        corr=many_pairwise_correlation(gt, test_mat[:, indices])
        gc.collect()
        all_corrs.append(corr)
    return all_corrs

def RSA(X,Y):

    temp=1-np.corrcoef(Y)
    temp2=1-np.corrcoef(X)
    temp = np.array(temp[np.triu_indices(temp.shape[0], k=1)])
    temp2 = np.array(temp2[np.triu_indices(temp2.shape[0], k=1)])
    return [kendalltau(temp,temp2)[0]]

def CKA(X,Y):

    X =X- X.mean(axis=0)
    Y =Y- Y.mean(axis=0)

    gc.collect()
    YTX = Y.T.dot(X)
    def traceATA(A):
        num_rows, _ = A.shape
        out=0
        for i in progress_bar(range(num_rows)):
            for j in range(num_rows):
                    out += np.sum(A[i] * A[j])**2
        return out
    
    return [(YTX ** 2).sum() / np.sqrt(traceATA(X)*traceATA(Y))]
        
        
        
        
        



def VERSA(X,Y):


    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        predictor = RidgeCV(alphas=np.logspace(-8,8,17))
        predictor.fit(normalize(X[train_idx]), Y[train_idx])
        
        y_pred = predictor.predict(normalize(X[test_idx]))
        model_rdm = 1 - np.corrcoef(y_pred)
        Y_rdms=1 - np.corrcoef(Y[test_idx])
        temp = np.array(model_rdm[np.triu_indices(model_rdm.shape[0], k=1)])
        temp2 = np.array(Y_rdms[np.triu_indices(Y_rdms.shape[0], k=1)])
        
        
        all_corrs.append(kendalltau(temp,temp2)[0])
        del predictor
        gc.collect()
        

    return all_corrs
    
def linearshapemetric(X,Y,alpha=1):
    def whiten(X, alpha=alpha, preserve_variance = True, eigval_tol=1e-7):
        if alpha > (1 - eigval_tol):
            return X, np.eye(X.shape[1])

        lam, V = np.linalg.eigh(X.T @ X)
        lam = np.maximum(lam, eigval_tol)


        d = alpha + (1 - alpha) * lam ** (-1 / 2)

        if preserve_variance:


            new_var = np.sum(
                (alpha ** 2) * lam
                + 2 * alpha * (1 - alpha) * (lam ** 0.5)
                + ((1 - alpha) ** 2) * np.ones_like(lam)
            )


            d *= np.sqrt(np.sum(lam) / new_var)

        Z = (V * d[None, :]) @ V.T

        return X @ Z, Z
    
    def partial_fit(X,alpha=alpha):
        mx = np.mean(X, axis=0)
        Xw, Zx = whiten(X - mx[None, :], alpha, preserve_variance=True)
        return (X,mx, Xw, Zx)
    
    def compute_distance(X, Y, X_test, Y_test):

        X,Y,X_test, Y_test=finalize_fit(partial_fit(X),partial_fit(Y),X_test, Y_test)

        dist_test = angular_distance(X_test, Y_test)
        return  dist_test
    def angular_distance(X, Y):
        normalizer = np.linalg.norm(X.ravel()) * np.linalg.norm(Y.ravel())
        corr = np.dot(X.ravel(), Y.ravel()) / normalizer
        return np.arccos(np.clip(corr, -1.0, 1.0))
    
    def finalize_fit(cache_X,cache_Y,X_test, Y_test):
        X,mx_, Xw, Zx = cache_X
        Y,my_, Yw, Zy = cache_Y
        U, _, Vt = scipy.linalg.svd(Xw.T @ Yw, lapack_driver='gesvd')
        Wx_ = Zx @ U
        Wy_ = Zy @ Vt.T

        return (X - mx_[None, :]) @ Wx_,(Y - my_[None, :]) @ Wy_,(X_test - mx_[None, :]) @ Wx_,(Y_test - my_[None, :]) @ Wy_


    n = min(X.shape[-1], Y.shape[-1], Y.shape[0])
 
    
    if X.shape[-1] != n:
        pca = PCA(n, random_state=42)
        X=pca.fit_transform(X)
    if Y.shape[-1] != n:
        pca = PCA(n, random_state=42)
        Y=pca.fit_transform(Y)

    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        D_test = compute_distance(X[train_idx],Y[train_idx], X[test_idx],Y[test_idx])
        all_corrs.append(D_test.sum())
    return all_corrs

def PLSreg(X,Y):
    all_corrs=[]
    kf = KFold(n_splits=5,  shuffle=True, random_state = 42)
    for i, (train_idx, test_idx) in enumerate(kf.split(X)):
        predictor = PLSRegression(n_components=25)
        predictor.fit(normalize(X[train_idx]), Y[train_idx])
                    
        y_pred = predictor.predict(normalize(X[test_idx]))
        
        corr=many_pairwise_correlation(y_pred,Y[test_idx])
        del predictor
        gc.collect()
        all_corrs.append(corr)

    return all_corrs
        
all_metrics={
             'LinearPredictivity':LinearPredictivity,
             'reverseLinearPredictivity':reverseLinearPredictivity,
             'PLSreg':PLSreg,
             'pairwisematching':pairwisematching,
             'SoftMatching':SoftMatching,
             'RSA':RSA,
             'VERSA':VERSA,
             'CKA':CKA,
             'LinearShapeMetric':linearshapemetric
             }        