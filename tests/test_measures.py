from functools import partial
import numpy as np

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


def test_equiv_lange_cka_rsa():
    cka = similarity.make("measure/repsim/cka-")
    rsa = similarity.make("measure/rsatoolbox/rsa-euclidean_centered_rdm-cosine")

    X = np.random.randn(3, 2)
    Y = np.random.randn(3, 2)

    print(cka(X, Y), rsa(X, Y))


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
        scores[k] = v(X, Y)
        print(k, scores[k])
    # print(scores)

    # bar plot
    import matplotlib.pyplot as plt
    plt.bar(scores.keys(), scores.values())
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    test_consistency()
    test_netrep()
    # test_measures()
    # test_equiv_lange_cka_rsa()