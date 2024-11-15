from functools import partial
import numpy as np
import rsatoolbox

import similarity

def test_measures():
    """
    Make sure can create all measures and they run without error.
    """

    # X = np.random.randn(15, 20, 30)
    # Y = np.random.randn(15, 20, 30)

    X = np.random.randn(40, 30)
    Y = np.random.randn(40, 30)

    measures = similarity.all_measures()
    print(len(measures))
    breakpoint()

    for measure_id in measures:
        print(f"Testing {measure_id}")
        # measure = similarity.make_measure(measure_id)
        measure = similarity.make(measure_id)

        score = measure(X, Y)
        print(f"score: {score}")
        print()


def centering(rdm):
    D = rdm.get_matrices()[0]
    print("centering, D", D)

    n = D.shape[0]
    C = np.eye(n) - np.ones([n, n]) / n
    D_centered = C @ D @ C
    # rdm = rsatoolbox.rdm.rdms.RDMs(
    #     dissimilarities=np.array([X_centered]),
    #     dissimilarity_measure=rdm.dissimilarity_measure,
    #     descriptors=rdm.descriptors,
    #     rdm_descriptors=rdm.rdm_descriptors,
    #     pattern_descriptors=rdm.pattern_descriptors
    # )
    print("centering, D_centered:", D_centered)
    print("------")
    import rsatoolbox
    rdm = rdm.copy()
    rdm.dissimilarities = rsatoolbox.util.rdm_utils.batch_to_vectors(D_centered[None])[0]
    return rdm

def compute_rsa(X, Y):
    """
    Helper function to compute representational similarity analysis (RSA) using the rsatoolbox library
    """
    X = rsatoolbox.data.Dataset(X)
    Y = rsatoolbox.data.Dataset(Y)

    rdm1 = rsatoolbox.rdm.calc_rdm(X, method="euclidean")
    rdm2 = rsatoolbox.rdm.calc_rdm(Y, method="euclidean")

    # problem is that rsatoolbox removes the diagonal when creating RDMs objects
    _rdm1 = centering(rdm1)
    _rdm2 = centering(rdm2)

    # sim = rsatoolbox.rdm.compare(_rdm1, _rdm2, method="cosine")
    # sim = sim[0][0]

    Dxx = _rdm1.get_matrices()[0]
    Dyy = _rdm2.get_matrices()[0]
    print("Dxx", Dxx)
    print("Dyy", Dyy)
    sim = np.trace(Dxx @ Dyy) / (np.linalg.norm(Dxx, ord="fro") * np.linalg.norm(Dyy, ord="fro"))


    Dx = rdm1.get_matrices()[0]
    Dy = rdm2.get_matrices()[0]
    print("Dx", Dx)
    print("Dy", Dy)
    n = Dx.shape[0]
    print("nn", n)
    C = np.eye(n) - np.ones([n, n]) / n
    Dx = C @ Dx @ C
    Dy = C @ Dy @ C
    print("centered Dx", Dx)
    print("centered Dy", Dy)
    s = np.trace(Dx @ Dy) / (np.linalg.norm(Dx, ord="fro") * np.linalg.norm(Dy, ord="fro"))
    print(sim, s)
    breakpoint()
    return sim



def test_equiv_cka_rsa():
    cka = similarity.make("measure/netrep/cka")
    rsa = similarity.make("measure/rsatoolbox/rsa-euclidean_centered_rdm-cosine")

    X = np.random.randn(3, 2)
    Y = np.random.randn(3, 2)

    Dx = np.sum((X[:, None] - X[None])**2, axis=-1)
    Dy = np.sum((Y[:, None] - Y[None])**2, axis=-1)
    print("D shape:", Dx.shape)
    n = X.shape[0]
    print("nn", n)
    C = np.eye(n) - np.ones([n, n]) / n
    Dx = C @ Dx @ C
    Dy = C @ Dy @ C

    rsa_score2 = np.trace(Dx @ Dy) / (np.linalg.norm(Dx, ord="fro") * np.linalg.norm(Dy, ord="fro"))

    import rsatoolbox
    rsa_score3 = compute_rsa(X, Y)


    cka_score = cka(X, Y)
    rsa_score = rsa(X, Y)
    print(cka_score, rsa_score, rsa_score2, rsa_score3)

    rsa2 = similarity.make("measure/thingsvision/rsa-euclidean_centered_rdm-cosine")

    print(rsa2(X, Y))
    breakpoint()

    # unbiased_cka = similarity.make("measure/thingsvision/cka_kernel_linear_unbiased")
    # print(unbiased_cka(X, Y))

    # test_measures()


def center_rows_columns(X):
    m, n = X.shape
    C_rows = np.eye(m) - np.ones([m, m]) / m
    C_cols = np.eye(n) - np.ones([n, n]) / n
    return C_rows @ X @ C_cols


def kernel_to_distance(K):
    # Dij = Kii + Kjj - 2Kij
    D = np.diag(K)[:, None] + np.diag(K)[None, :] - 2 * K
    return D


def test_equiv_thingsvision_cka_rsa():
    X, Y = np.random.randn(30, 20), np.random.randn(30, 20)

    cka = similarity.make("measure/thingsvision/cka-kernel=linear-hsic=gretton-score")

    Dx_np = np.sum((X[:, None] - X[None])**2, axis=-1)
    Dy_np = np.sum((Y[:, None] - Y[None])**2, axis=-1)

    # TODO: automatically derive "rdm/*/squared_euclidean"
    Dx = similarity.make("rdm/thingsvision/euclidean")(X)**2
    Dy = similarity.make("rdm/thingsvision/euclidean")(Y)**2
    Dx_bis = similarity.make("rdm/thingsvision/squared_euclidean")(X)
    assert np.allclose(Dx, Dx_bis)


    Dx = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(X)
    Dy = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(Y)


    # X_test = center_rows_columns(X)
    X_test = similarity.make("preprocessing/center_rows_columns")(X)
    print(np.sum(X_test, axis=1))
    print(np.sum(X_test, axis=0))

    Dx_c = center_rows_columns(Dx)
    Dy_c = center_rows_columns(Dy)

    # n = X.shape[0]
    # C = np.eye(n) - np.ones([n, n]) / n
    # Dx_c = C @ Dx @ C
    # Dy_c = C @ Dy @ C

    rsa_score = np.trace(Dx_c @ Dy_c) / (np.linalg.norm(Dx_c, ord="fro") * np.linalg.norm(Dy_c, ord="fro"))

    # angular distance is the arccos of the cosine similarity
    rsa_score2 = similarity.make("distance/netrep/angular")(Dx_c, Dy_c)
    rsa_score2 = np.cos(rsa_score2)

    print("cka:", cka(X, Y), "rsa:", rsa_score, "rsa2:", rsa_score2)

    sum_sq = np.sum(X**2, axis=1, keepdims=True)
    Dx_np2 = sum_sq + sum_sq.T - 2 * np.dot(X, X.T)

    Dx = similarity.make("rdm/thingsvision/euclidean")(X)**2
    rdmX = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(X)
    rdmX = rdmX * X.shape[1]
    rdmX_bis = similarity.make("rdm/rsatoolbox/squared_euclidean")(X)
    assert np.allclose(rdmX, rdmX_bis)

    Dy = similarity.make("rdm/thingsvision/euclidean")(Y)**2
    rdmY = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(Y)
    rdmY = rdmY * Y.shape[1]

    Kx = similarity.make("kernel/thingsvision/linear")(X)
    D_Kx = kernel_to_distance(Kx)
    # print("Dx[0]", Dx[0])
    # print("rdmX[0]", rdmX[0])
    # print("D_Kx[0]", D_Kx[0])


    # derive rsa from cka
    Kx = similarity.make("kernel/thingsvision/linear")(X)
    Ky = similarity.make("kernel/thingsvision/linear")(Y)

    # can convert kernel to distance (+ normalized)
    D_Kx = kernel_to_distance(Kx)
    D_Ky = kernel_to_distance(Ky)

    D_kx_normalized = D_Kx / X.shape[1]
    D_ky_normalized = D_Ky / Y.shape[1]

    # can automatically convert angular distance to cosine similarity and vice versa
    rsa_score2 = similarity.make("distance/netrep/angular")(D_kx_normalized, D_ky_normalized)
    rsa_score2 = np.cos(rsa_score2)

    rsa_score = similarity.make("measure/rsatoolbox/rsa-rdm=euclidean-compare=cosine")(X, Y)
    print("rsa_score", rsa_score)
    print("rsa_score2", rsa_score2)


    cka_lange = similarity.make("measure/repsim/cka-kernel=linear-hsic=lange-score")(X, Y)


    Dx_nodiag = Dx - np.diag(np.diag(Dx))
    Dy_nodiag = Dy - np.diag(np.diag(Dy))
    s = similarity.make("distance/netrep/angular")(Dx_nodiag, Dy_nodiag)
    s = np.cos(s)
    s_bis = similarity.make("measure/netrep/cosine")(Dx_nodiag, Dy_nodiag)
    assert np.allclose(s, s_bis)

    Dx_c_nodiag = Dx_c - np.diag(np.diag(Dx_c))
    Dy_c_nodiag = Dy_c - np.diag(np.diag(Dy_c))
    s2 = similarity.make("distance/netrep/angular")(Dx_c_nodiag, Dy_c_nodiag)
    s2 = np.cos(s2)
    print("cka_lange", cka_lange, "s", s, "s2", s2)


    # derive lange cka from netrep cka
    Kx = similarity.make("kernel/netrep/linear")(X)
    Ky = similarity.make("kernel/netrep/linear")(Y)
    # center
    Kx = similarity.make("preprocessing/center_rows_columns")(Kx)
    Ky = similarity.make("preprocessing/center_rows_columns")(Ky)

    Kx_bis = similarity.make("kernel/netrep/linear-centered")(X)
    Ky_bis = similarity.make("kernel/netrep/linear-centered")(Y)
    assert np.allclose(Kx, Kx_bis)
    assert np.allclose(Ky, Ky_bis)

    # remove diagonal
    Kx = Kx - np.diag(np.diag(Kx))
    Ky = Ky - np.diag(np.diag(Ky))

    Kx_bis = similarity.make("kernel/netrep/linear-centered-zero_diagonal")(X)
    Ky_bis = similarity.make("kernel/netrep/linear-centered-zero_diagonal")(Y)
    assert np.allclose(Kx, Kx_bis)
    assert np.allclose(Ky, Ky_bis)

    s = similarity.make("measure/netrep/cosine")(Kx, Ky)
    cka_lange2 = similarity.make("measure/repsim/cka-kernel=linear-hsic=lange-score")(X, Y)
    print("cka_lange2", cka_lange2, "s", s)

    cka_lange3 = similarity.make("measure/netrep/cka-kernel=linear-hsic=lange-score")(X, Y)
    print("cka_lange3", cka_lange3)

    s2 = similarity.make("measure/rsatoolbox/zero_diagonal-cosine")(Kx, Ky)
    print("s2", s2)

    # verify rsatoolbox consistency
    rdmX = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(X)
    rdmY = similarity.make("rdm/rsatoolbox/squared_euclidean_normalized")(Y)
    s = similarity.make("measure/rsatoolbox/zero_diagonal-cosine")(rdmX, rdmY)

    datasetX = rsatoolbox.data.Dataset(X)
    datasetY = rsatoolbox.data.Dataset(Y)
    rdmX2 = rsatoolbox.rdm.calc_rdm(datasetX, method="euclidean")
    rdmY2 = rsatoolbox.rdm.calc_rdm(datasetY, method="euclidean")
    assert np.allclose(rdmX2.get_matrices()[0], rdmX)
    assert np.allclose(rdmY2.get_matrices()[0], rdmY)

    rX = rsatoolbox.rdm.RDMs(rdmX2.get_matrices()[0][None])
    rY = rsatoolbox.rdm.RDMs(rdmY2.get_matrices()[0][None])
    print(rX.get_matrices()[0].shape)
    print(rdmX2.get_matrices()[0].shape)
    assert np.allclose(rX.get_matrices()[0], rdmX2.get_matrices()[0])
    assert np.allclose(rY.get_matrices()[0], rdmY2.get_matrices()[0])

    s2 = rsatoolbox.rdm.compare(rdmX2, rdmY2, method="cosine")[0][0]

    gt = similarity.make("measure/rsatoolbox/rsa-rdm=euclidean-compare=cosine")(X, Y)


    rdmX = similarity.make("rdm/netrep/squared_euclidean")(X)
    rdmY = similarity.make("rdm/netrep/squared_euclidean")(Y)
    s3 = similarity.make("measure/netrep/cosine")(rdmX, rdmY)

    print("gt", gt, "s", s, "s2", s2, "s3", s3)
    assert np.allclose(gt, s)
    assert np.allclose(gt, s2)
    breakpoint()



    # cka: rsa, rsm=eucl, comp=cosine, center rdm (don't remove diag)
    # 0.44

    # cka lange: rsa, rsm=eucl, comp=cosine, center rdm, remove diag
    # 0.07
    # repsim: zero the diagonal after centering, so the kernel matrix is not centered for unbiased cka!

    # rsatoolbox eucl cosine: rsa, rsm=eucl, comp=cosine, remove diag, normalize rdm, don't center rdm
    # 0.91


def test_netrep():
    netrep_measures = similarity.make("measure/netrep/*")
    X = np.random.randn(30, 20)
    Y = np.random.randn(30, 20)
    print(list(netrep_measures.keys()))
    alpha = 1
    for k, v in netrep_measures.items():
        if "{alpha}" in k:
            v = partial(v, alpha=alpha)
        print(k)
        print(v(X, Y))
    breakpoint()


def test_consistency():
    measure_name = "cka"
    measure = similarity.make(f"measure/*/{measure_name}")
    X, Y = np.random.randn(30, 20), np.random.randn(30, 20)
    scores = {}
    for k, v in measure.items():
        print(k)
        scores[k] = v(X, Y)
        print(k, scores[k])
    # print(scores)

    # bar plot
    import matplotlib.pyplot as plt
    plt.bar(scores.keys(), scores.values())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


def test_rdm_kernel():
    X = np.random.randn(30, 20)
    Y = np.random.randn(30, 20)

    K1 = similarity.make("kernel/netrep/linear")(X)
    K2 = similarity.make("kernel/rsatoolbox/linear")(X)
    assert np.allclose(K1, K2)

    D1 = similarity.make("rdm/netrep/euclidean")(X)
    D2 = similarity.make("rdm/rsatoolbox/euclidean")(X)
    assert np.allclose(D1, D2)

    K1_centered = similarity.make("kernel/netrep/linear-centered")(X)
    K2_centered = similarity.make("kernel/rsatoolbox/linear-centered")(X)
    assert np.allclose(K1_centered, K2_centered)

    print("Test RDM and kernel equivalence passed")


    cka_lange = similarity.make("measure/netrep/cka-kernel=linear-hsic=lange-score")
    cka_rsatoolbox = similarity.make("measure/rsatoolbox/cka-kernel=linear-hsic=lange-score")

    print(cka_lange(X, Y), cka_rsatoolbox(X, Y))
    breakpoint()


if __name__ == "__main__":
    # fix numpy random seed
    np.random.seed(5)
    # test_rdm_kernel()

    # test_consistency()
    # test_netrep()

    test_equiv_thingsvision_cka_rsa()

    # test_measures()
    # test_equiv_lange_cka_rsa()