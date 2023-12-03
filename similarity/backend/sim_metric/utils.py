"""
Code taken from sim_metroc/dists/score_pair.py and packaged into functions
"""
import numpy as np
import similarity.backend.sim_measure.dists.scoring as scoring


def row_centering(rep1, rep2):
    # center each row
    rep1 = rep1 - rep1.mean(axis=1, keepdims=True)
    rep2 = rep2 - rep2.mean(axis=1, keepdims=True)
    return rep1, rep2


def normalize(rep1, rep2):
    # normalize each representation
    rep1 = rep1 / np.linalg.norm(rep1)
    rep2 = rep2 / np.linalg.norm(rep2)
    return rep1, rep2


def pwcca_dist(rep1, rep2):
    cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = scoring.cca_decomp(
        rep1, rep2
    )
    return scoring.pwcca_dist(rep1, cca_rho, transformed_rep1)


def mean_sq_cca_corr(rep1, rep2):
    cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = scoring.cca_decomp(
        rep1, rep2
    )
    return scoring.mean_sq_cca_corr(cca_rho)


def mean_cca_corr(rep1, rep2):
    cca_u, cca_rho, cca_vh, transformed_rep1, transformed_rep2 = scoring.cca_decomp(
        rep1, rep2
    )
    return scoring.mean_cca_corr(cca_rho)
