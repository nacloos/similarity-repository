import rsatoolbox


def compute_rsa(X, Y, rdm_method, compare_method):
    X = rsatoolbox.data.Dataset(X)
    Y = rsatoolbox.data.Dataset(Y)

    rdm1 = rsatoolbox.rdm.calc_rdm(X, method=rdm_method)
    rdm2 = rsatoolbox.rdm.calc_rdm(Y, method=rdm_method)
    sim = rsatoolbox.rdm.compare(rdm1, rdm2, method=compare_method)
    sim = sim[0][0]
    return sim
