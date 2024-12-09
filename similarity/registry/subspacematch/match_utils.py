# Taken from https://github.com/MeckyWu/subspace-match/blob/master/utils/match_utils.py
import numpy as np
# import time

# _DEBUG = False
_ZERO = 1e-16


def dist2plane(x, Y):
    """
    Calucating the distance of a vector to a plane.
    Assume that the norm of x is large enough
    :param X: [M,]
    :param Y: [N2, D]
    :return: (scalar)
    """
    assert x.ndim == 1
    x_norm = np.linalg.norm(x)
    x = x.reshape(-1, 1)
    Y_t = np.transpose(Y)
    if x_norm < _ZERO:
        return 0.
    solution = np.linalg.lstsq(Y_t, x, rcond=None)
    dist = np.linalg.norm(np.dot(Y_t, solution[0]) - x)
    return dist / x_norm


def dists2plane(X, Y):
    """
    Calucating the distances of a group of vectors to a plane
    Assume norm is large enough
    :param X: [N1, D]
    :param Y: [N2, D]
    :return: [N1,]
    """
    Y_t = np.transpose(Y)
    X_t = np.transpose(X)
    solution = np.linalg.lstsq(Y_t, X_t, rcond=None)
    dist = np.linalg.norm(np.dot(Y_t, solution[0]) - X_t, axis=0)
    norm = np.linalg.norm(X_t, axis=0)
    return dist / norm


def remove_zeros(X):
    """
    Remove zero-norm vectors
    Args:
        X: [N, D]

    Returns:
        non-zero vectors: [N',]
        non-zero indices: [N',]

    """
    assert X.ndim == 2, "Only support 2-D X"
    norm_X = np.linalg.norm(X, axis=1)
    non_zero = np.where(norm_X > _ZERO)[0]
    return X[non_zero], non_zero


def find_maximal_epsilon(X, Y):
    """
    Find maximal distance between X and Y
    Args:
        X: [N1, D]
        Y: [N2, D]

    Returns:
        scalar
    """
    X, idx_X = remove_zeros(X)
    Y, idx_Y = remove_zeros(Y)

    res = 0
    if len(idx_X) > 0 and len(idx_Y) > 0:
        dist_X2Y = dists2plane(X, Y)
        dist_Y2X = dists2plane(Y, X)
        max_dist_X2Y = np.max(dist_X2Y)
        max_dist_Y2X = np.max(dist_Y2X)
        res = max(res, max_dist_X2Y, max_dist_Y2X)

    return res


def find_minimal_epsilon(X, Y, L, R, min_size=0, tol=1e-4):
    """
    Find minimal epsilon such that X has a match set with Y whose size is larger than min size
    Args:
        X: [N1, D]
        Y: [N2, D]
        L: left border, no match
        R: right border, match size >= min_size
        min_size: minimal size of match set
        tol: tolerance for algorithms

    Returns:
        scalar: minimal epsilon
        idx_X: X's match set
        idx_Y: Y's match set

    """
    idx_X = np.arange(X.shape[0])
    idx_Y = np.arange(Y.shape[0])

    while R - L > tol:
        M = (L + R) * 0.5
        idxx, idxy = find_maximal_match(X[idx_X], Y[idx_Y], M)
        if min(len(idxx), len(idxy)) >= min_size:
            R = M
            idx_X = idx_X[idxx]
            idx_Y = idx_Y[idxy]
        else:
            L = M

    return R, idx_X, idx_Y


def find_maximal_match(X, Y, eps, has_purge=False):
    """
    Find maximal match set between X and Y
    Args:
        X: [N1, D]
        Y: [N2, D]
        eps: scalar
        has_purge: whether X and Y have removed zero vectors

    Returns:
        idx_X: X's match set indices
        idx_Y: Y's match set indices
    """
    assert X.ndim == 2 and Y.ndim == 2, 'Check dimensions of X and Y'
    # if _DEBUG: print('eps={:.4f}'.format(eps))

    if not has_purge:
        X, non_zero_X = remove_zeros(X)
        Y, non_zero_Y = remove_zeros(Y)

    idx_X = np.arange(X.shape[0])
    idx_Y = np.arange(Y.shape[0])

    if len(idx_X) == 0 or len(idx_Y) == 0:
        return idx_X[[]], idx_Y[[]]

    flag = True
    while flag:
        flag = False

        # tic = time.time()
        dist_X = dists2plane(X[idx_X], Y[idx_Y])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_X = idx_X[dist_X <= eps]

        if len(remain_idx_X) < len(idx_X):
            flag = True

        idx_X = remain_idx_X
        if len(idx_X) == 0:
            idx_Y = idx_Y[[]]
            break

        # tic = time.time()
        dist_Y = dists2plane(Y[idx_Y], X[idx_X])
        # toc = time.time()
        # print(toc-tic)
        remain_idx_Y = idx_Y[dist_Y <= eps]

        if len(remain_idx_Y) < len(idx_Y):
            flag = True

        idx_Y = remain_idx_Y
        if len(idx_Y) == 0:
            idx_X = idx_X[[]]
            break

        # if _DEBUG: print('|X|={:d}, |Y|={:d}'.format(len(idx_X), len(idx_Y)))

    if not has_purge:
        idx_X = non_zero_X[idx_X]
        idx_Y = non_zero_Y[idx_Y]

    return idx_X, idx_Y


def find_minimal_match(X, Y, set_idx, v_idx, eps, is_max=False, is_random=False):
    """
    Find v-minimal match set
    Args:
        X: [N1, D]
        Y: [N2, D]
        set_idx: 0 for X and 1 for Y
        v_idx: index of v
        eps: epsilon of interests
        is_max: whether X and Y is maximal matching set
        is_random: whether to use the randomized method

    Returns:
        A simple match: a tuple or None
    """

    # if _DEBUG: print('eps={:.4f}'.format(eps))

    uncheck_X = np.ones([X.shape[0]], dtype=bool)
    uncheck_Y = np.ones([Y.shape[0]], dtype=bool)

    idx_X = np.arange(X.shape[0])
    idx_Y = np.arange(Y.shape[0])
    if not is_max:
        idx_X, idx_Y = find_maximal_match(X, Y, eps)

    if set_idx == 0:
        if v_idx in idx_X:
            uncheck_X[v_idx] = False
        else:
            return None

    if set_idx == 1:
        if v_idx in idx_Y:
            uncheck_Y[v_idx] = False
        else:
            return None

    # tally = 0
    while True:
        res_X = np.where(uncheck_X[idx_X])[0]
        res_Y = np.where(uncheck_Y[idx_Y])[0]

        _idx_X = idx_X.copy()
        _idx_Y = idx_Y.copy()

        if is_random:
            if np.random.rand() > 0.5:
                if len(res_X) > 0:
                    res_X = np.random.choice(res_X)
                    uncheck_X[idx_X[res_X]] = False
                    _idx_X = np.delete(_idx_X, res_X)
                elif len(res_Y) > 0:
                    res_Y = np.random.choice(res_Y)
                    uncheck_Y[idx_Y[res_Y]] = False
                    _idx_Y = np.delete(_idx_Y, res_Y)
                else:
                    break
            else:
                if len(res_Y) > 0:
                    res_Y = np.random.choice(res_Y)
                    uncheck_Y[idx_Y[res_Y]] = False
                    _idx_Y = np.delete(_idx_Y, res_Y)
                elif len(res_X) > 0:
                    res_X = np.random.choice(res_X)
                    uncheck_X[idx_X[res_X]] = False
                    _idx_X = np.delete(_idx_X, res_X)
                else:
                    break
        else:
            if len(res_X) > 0:
                res_X = res_X[0]
                uncheck_X[idx_X[res_X]] = False
                _idx_X = np.delete(_idx_X, res_X)
            elif len(res_Y) > 0:
                res_Y = res_Y[0]
                uncheck_Y[idx_Y[res_Y]] = False
                _idx_Y = np.delete(_idx_Y, res_Y)
            else:
                break

        # tic = time.time()
        idxx, idxy = find_maximal_match(X[_idx_X], Y[_idx_Y], eps, has_purge=True)
        # toc = time.time()
        # print(toc - tic)

        if set_idx == 0 and (v_idx not in _idx_X[idxx]):
            continue
        if set_idx == 1 and (v_idx not in _idx_Y[idxy]):
            continue

        idx_X = _idx_X
        idx_Y = _idx_Y
        # print('update %d, %d and %d' % (tally, len(idx_X), len(idx_Y)))
        # tally += 1

    return idx_X, idx_Y


def unittest():
    eps = 0.1
    np.random.seed(0)

    # check distance function
    print(dist2plane(np.array([1,1]), np.array([[1,0]])))
    print(dists2plane(np.array([[1,1]]), np.array([[1,0]])))

    # check remove_zeros
    X = np.array([[1, 0, 0, 0, 0, 0],
                  [0, 0, 0, 0, 0, 0.5* _ZERO],
                  ])
    print(remove_zeros(X))

    # check find maximal and minimum match
    X = np.array([[1, 0.5, 0, 0, 0, 0],
                  [0, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  ])
    Y = np.array([[0, 0, 1, 0, 0, 0],
                  [1, 0, 0, 0, 0, 0],
                  [0.5, 1, 0, 0, 0, 0],
                  [0, 0, 0, 1, 0, 0],
                  [0, 0, 0, 0, 1, 0],
                  ])

    print(find_maximal_epsilon(X[0:1], Y[0:1]))
    print(find_maximal_match(X, Y, eps))
    print(find_minimal_match(X, Y, 0, 0, eps))

    from numpy.linalg.linalg import matmul
    U = np.random.randn(10, 200)
    X1 = matmul(np.random.randn(20, 10), U)
    Y1 = matmul(np.random.randn(20, 10), U)
    X = np.random.randn(100, 200)
    X = np.vstack([X1,X])
    Y = np.random.randn(100, 200)
    Y = np.vstack([Y1,Y])

    print(find_maximal_match(X, Y, eps))
    print(find_minimal_match(X, Y, 0, 0, eps))
    print(find_maximal_epsilon(X, Y))


if __name__ == '__main__':
    unittest()
