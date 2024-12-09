from .match_utils import find_maximal_epsilon, find_maximal_match


def maximum_match_score(X, Y, epsilon):
    # Adapted from https://github.com/MeckyWu/subspace-match/blob/master/calc_max_match.py
    idx_X, idx_Y = find_maximal_match(X, Y, epsilon)
    mms = float(len(idx_X) + len(idx_Y)) / (len(X) + len(Y))
    return mms
