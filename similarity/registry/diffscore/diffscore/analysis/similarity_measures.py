from functools import partial
import numpy as np
from sklearn.model_selection import KFold

import torch

# import diffscore
# from diffscore import register, make


measures = {}

def register(id, function=None):
    def _register(id, function):
        measures[id] = function

    if function is None:
        def decorator(function):
            _register(id, function)
            return function
        return decorator
    else:
        _register(id, function)

def make(id):
    return measures[id]


@register("transform/zero_diagonal")
def zero_diagonal(X):
    return X - torch.diag(torch.diag(X))


@register("transform/center_rows_columns")
def center_rows_columns(X):
    def center_rows(X):
        return X - torch.mean(X, dim=1, keepdim=True)
    def center_cols(X):
        return X - torch.mean(X, dim=0, keepdim=True)
    return center_cols(center_rows(X))


# TODO: whiten, derive cca from cka by first whitening data (can we also derive R2 linreg from cka?)
@register("transform/whiten")
def whiten(X, alpha=0, preserve_variance=True, eigval_tol=1e-7):
    # Taken from https://github.com/ahwillia/netrep/blob/0186b8a77ec1ebaf541cc8f7173cb2556df8a8f0/netrep/utils.py#L178C1-L255C20
    # Return early if regularization is maximal (no whitening).
    if alpha > (1 - eigval_tol):
        return X, torch.eye(X.shape[1])

    # Compute eigendecomposition of covariance matrix
    lam, V = torch.linalg.eigh(X.T @ X)
    lam = torch.maximum(lam, torch.tensor(eigval_tol))

    # Compute diagonal of (partial) whitening matrix.
    #
    # When (alpha == 1), then (d == ones).
    # When (alpha == 0), then (d == 1 / sqrt(lam)).
    d = alpha + (1 - alpha) * lam ** (-1 / 2)

    # Rescale the whitening matrix.
    if preserve_variance:
        # Compute the variance of the transformed data.
        #
        # When (alpha == 1), then new_var = sum(lam)
        # When (alpha == 0), then new_var = len(lam)
        new_var = torch.sum(
            (alpha ** 2) * lam
            + 2 * alpha * (1 - alpha) * (lam ** 0.5)
            + ((1 - alpha) ** 2) * torch.ones_like(lam)
        )

        # Now re-scale d so that the variance of (X @ Z)
        # will equal the original variance of X.
        d *= torch.sqrt(torch.sum(lam) / new_var)

    # Form (partial) whitening matrix.
    Z = (V * d[None, :]) @ V.T

    # An alternative regularization strategy would be:
    #
    # lam, V = np.linalg.eigh(X.T @ X)
    # d = lam ** (-(1 - alpha) / 2)
    # Z = (V * d[None, :]) @ V.T

    # Returned (partially) whitened data and whitening matrix.
    # return X @ Z, Z
    return X @ Z

@register("transform/kernel_to_rdm")
def kernel_to_rdm(kernel):
    def _rdm(X):
        K = kernel(X)
        return torch.diag(K)[:, None] + torch.diag(K)[None, :] - 2 * K
    return _rdm


@register("transform/distance_to_similarity")
def distance_to_similarity(distance):
    def _similarity(X, Y):
        norm_X = distance(X, torch.zeros_like(X))
        norm_Y = distance(Y, torch.zeros_like(Y))
        sim = 1/2 * (norm_X**2 + norm_Y**2 - distance(X, Y)**2)
        sim /= (norm_X * norm_Y)
        return sim
    return _similarity


@register("kernel/linear")
def kernel_linear(X):
    return X @ X.T


# TODO: derive it from distance/euclidean (so that define only distance)
# @register("measure/cosine")
# def measure_cosine(X, Y):
#     return (X.ravel() @ Y.ravel()) / (torch.linalg.norm(X, ord='fro') * torch.linalg.norm(Y, ord='fro'))


@register("rdm/correlation")
def rdm_correlation(X):
    X_centered = X - X.mean(dim=1, keepdim=True)
    X_centered = X_centered / torch.sqrt(torch.sum(X_centered**2, dim=1, keepdim=True))
    return 1 - X_centered @ X_centered.T


@register("distance/alpha_procrustes")
def distance_alpha_procrustes(X, Y, alpha):
    """
    https://arxiv.org/pdf/1908.09275
    """
    # implementation adapted from https://github.com/rsagroup/rsatoolbox/blob/c23f41224a50326dfbd5675b284dfc129bea5e8a/src/rsatoolbox/rdm/compare.py#L678
    sX, uX = torch.linalg.eigh(X)
    sY, uY = torch.linalg.eigh(Y)
    sX = torch.clip(sX, min=0.0)
    sY = torch.clip(sY, min=0.0)
 
    X_alpha = uX @ (sX[:, None]**alpha * uX.T)
    Y_alpha = uY @ (sY[:, None]**alpha * uY.T)

    X_alpha_2 = X_alpha @ X_alpha
    Y_alpha_2 = Y_alpha @ Y_alpha

    d = 1/alpha * torch.sqrt(torch.clip(
        torch.trace(X_alpha_2) + torch.trace(Y_alpha_2)
        - 2*torch.sum(torch.sqrt(torch.clip(torch.linalg.eigvalsh(X_alpha @ Y_alpha_2 @ X_alpha), min=0.0))),
        min=0.0
    ))
    return d


@register("distance/bures")
def distance_bures(X, Y):
    # TODO: take distance(X, 0) in distance_to_similarity
    if torch.all(X == 0):
        return torch.sqrt(torch.trace(Y))
    if torch.all(Y == 0):
        return torch.sqrt(torch.trace(X))
    # implement with the nuclear norm? more efficient?
    # sXY = torch.linalg.svdvals(X.T @ Y)  # nuclear norm ||X.T Y||_*
    # breakpoint()
    # return torch.sqrt(torch.trace(X) + torch.trace(Y) - 2 * torch.sum(sXY))
    sX, uX = torch.linalg.eigh(X)
    X_sqrt = uX @ (torch.sqrt(torch.clip(sX[:, None], min=0.0)) * uX.T)

    return torch.sqrt(
        torch.trace(X) + torch.trace(Y)
        - 2 * torch.sum(torch.sqrt(torch.clip(torch.linalg.eigvalsh(X_sqrt @ Y @ X_sqrt), min=0.0)))
    )

@register("similarity/bures")
def similarity_bures(X, Y):
    # va, ua = np.linalg.eigh(A)
    # Asq = ua @ (np.sqrt(np.maximum(va[:, None], 0.0)) * ua.T)
    # num = np.sum(np.sqrt(np.maximum(np.linalg.eigvalsh(Asq @ B @ Asq), 0.0)))
    # denom = np.sqrt(np.trace(A) * np.trace(B))
    # return num / denom

    sX, uX = torch.linalg.eigh(X)
    X_sqrt = uX @ (torch.sqrt(torch.clip(sX[:, None], min=0.0)) * uX.T)
    num = torch.sum(torch.sqrt(torch.clip(torch.linalg.eigvalsh(X_sqrt @ Y @ X_sqrt), min=0.0)))

    # sY, uY = torch.linalg.eigh(Y)
    # denom = torch.sqrt(torch.sum(sY) * torch.sum(sY))
    # TODO: trace(X) gives 0
    denom = torch.sqrt(torch.trace(X) * torch.trace(Y))
    return num / denom


@register("distance/euclidean")
def distance_euclidean(X, Y):
    return torch.linalg.norm(X - Y, ord='fro')


# TODO: rename "centered" by "row_col_centered"? e.g. for cka
@register("similarity/triu_centered-zero_diagonal-cosine")
# TODO: better name for this?  e.g. "similarity/triu_centered-zero_diagonal-cosine"
@register("similarity/upper_triangular_correlation")
def similarity_upper_triangular_correlation(X, Y):
    # https://rsatoolbox.readthedocs.io/en/stable/comparing.html#pearson-correlation
    n = X.shape[0]
    X = X - n / (n - 1) * X.mean()
    Y = Y - n / (n - 1) * Y.mean()
    return make("similarity/zero_diagonal-cosine")(X, Y)


def embedding_similarity(X, Y, embedding, similarity):
    # embedding: often a kernel or a rdm function
    return similarity(embedding(X), embedding(Y))



# @register("measure/nbs-score")
# def nbs(X, Y):
#     Kx = make("kernel/linear")(X)
#     Ky = make("kernel/linear")(Y)
#     Kx_centered = make("transform/center_rows_columns")(Kx)
#     Ky_centered = make("transform/center_rows_columns")(Ky)
#     score = make("similarity/bures")(Kx_centered, Ky_centered)
#     return score


# @register("measure/procrustes-angular_score")
# def procrustes(X, Y):
#     Kx = make("kernel/linear")(X)
#     Ky = make("kernel/linear")(Y)
#     Kx_centered = make("transform/center_rows_columns")(Kx)
#     Ky_centered = make("transform/center_rows_columns")(Ky)
#     score = make("similarity/bures")(Kx_centered, Ky_centered)
#     score = 1 - 2 / np.pi * torch.arccos(score)
#     return score


def derive_methods_once(registry):
    derived = {}

    # centered, whitened kernel
    kernel_ids = [id for id in registry.keys() if id.startswith("kernel/")]
    for kernel_id in kernel_ids:
        name = kernel_id.split("/")[-1]

        if "centered-" not in name:
            def _center(X, kernel, **kwargs):
                X_centered = X - X.mean(dim=0, keepdim=True)
                return kernel(X_centered, **kwargs)

            new_id = f"kernel/centered-{name}"
            derived[new_id] = partial(_center, kernel=registry[kernel_id])

        if "whitened-" not in name:
            def _whiten(X, kernel, **kwargs):
                X_whitened = make("transform/whiten")(X, **kwargs)
                return kernel(X_whitened, **kwargs)

            new_id = f"kernel/whitened-{name}"
            derived[new_id] = partial(_whiten, kernel=registry[kernel_id])


    # distance to similarity
    distance_ids = [id for id in registry.keys() if id.startswith("distance/")]
    for distance_id in distance_ids:
        name = distance_id.split("/")[-1]
        if name == "euclidean":
            name = "cosine"  # euclidean distance <-> cosine similarity
        sim_id = f"similarity/{name}"

        dist_function = registry[distance_id]
        derived[sim_id] = distance_to_similarity(dist_function)


    # center, zero_diagonal similarity
    sim_ids = [id for id in registry.keys() if id.startswith("similarity/")]
    for sim_id in sim_ids:
        name = sim_id.split("/")[-1]

        if "zero_diagonal-" not in name:
            def _zero_diagonal_similarity(X, Y, similarity, **kwargs):
                return similarity(
                    make("transform/zero_diagonal")(X),
                    make("transform/zero_diagonal")(Y),
                    **kwargs
                )
            derived[f"similarity/zero_diagonal-{name}"] = partial(_zero_diagonal_similarity, similarity=registry[sim_id])
        
        if "centered-" not in name:
            def _centered_similarity(X, Y, similarity, **kwargs):
                return similarity(
                    make("transform/center_rows_columns")(X),
                    make("transform/center_rows_columns")(Y),
                    **kwargs
                )
            derived[f"similarity/centered-{name}"] = partial(_centered_similarity, similarity=registry[sim_id])


    # derive kernel and RDM similarity measures
    kernel_ids = [id for id in registry.keys() if id.startswith("kernel/")]
    rdm_ids = [id for id in registry.keys() if id.startswith("rdm/")]
    sim_ids = [id for id in registry.keys() if id.startswith("similarity/")]
    for sim_id in sim_ids:
        sim_name = sim_id.split("/")[-1]
        # add parentheses to separate the measure's args from the similarity's args
        sim_name = f"({sim_name})" if "-" in sim_name else sim_name
 
        for kernel_id in kernel_ids:
            kernel_name = kernel_id.split("/")[-1]
            kernel_name = f"({kernel_name})" if "-" in kernel_name else kernel_name
            new_id = f"measure/kernel={kernel_name}-similarity={sim_name}-score"
            derived[new_id] = partial(embedding_similarity, embedding=registry[kernel_id], similarity=registry[sim_id])

        for rdm_id in rdm_ids:
            rdm_name = rdm_id.split("/")[-1]
            rdm_name = f"({rdm_name})" if "-" in rdm_name else rdm_name
            new_id = f"measure/rdm={rdm_name}-similarity={sim_name}-score"
            derived[new_id] = partial(embedding_similarity, embedding=registry[rdm_id], similarity=registry[sim_id])


    # measure/*-score to measure/*-angular_score
    for id in registry.keys():
        if "measure/" in id and id.endswith("-score"):
            def _angular_score(X, Y, measure, **kwargs):
                score = measure(X, Y, **kwargs) 
                return 1 - 2 / torch.pi * torch.arccos(score)

            new_id = id.replace("-score", "-angular_score")
            derived[new_id] = partial(_angular_score, measure=registry[id])

        if "measure/" in id and id.endswith("-angular_score"):
            def _score(X, Y, measure, **kwargs):
                score = measure(X, Y, **kwargs) 
                return torch.cos(torch.pi/2 * (1 - score))

            new_id = id.replace("-angular_score", "-score")
            derived[new_id] = partial(_score, measure=registry[id])


    # kernel similarity measures with existing names
    if "measure/kernel=linear-similarity=(centered-cosine)-score" in registry:
        derived["measure/cka-kernel=linear-hsic=gretton-score"] = registry["measure/kernel=linear-similarity=(centered-cosine)-score"]

    if "measure/kernel=linear-similarity=(centered-zero_diagonal-cosine)-score" in registry:
        derived["measure/cka-kernel=linear-hsic=lange-score"] = registry["measure/kernel=linear-similarity=(centered-zero_diagonal-cosine)-score"]

    if "measure/kernel=linear-similarity=(centered-bures)-score" in registry:
        derived["measure/nbs"] = registry["measure/kernel=linear-similarity=(centered-bures)-score"]

    if "measure/kernel=linear-similarity=(centered-bures)-angular_score" in registry:
        # different names for the same measure
        derived["measure/nbs-angular_score"] = registry["measure/kernel=linear-similarity=(centered-bures)-angular_score"]
        derived["measure/procrustes-angular_score"] = registry["measure/kernel=linear-similarity=(centered-bures)-angular_score"]

    if "measure/kernel=(centered-whitened-linear)-similarity=bures-score" in registry:
        derived["measure/cca-score"] = registry["measure/kernel=(centered-whitened-linear)-similarity=bures-score"]

    if "measure/kernel=(centered-whitened-linear)-similarity=cosine-score" in registry:
        derived["measure/cca-squared_score"] = registry["measure/kernel=(centered-whitened-linear)-similarity=cosine-score"]

    # kernel to rdm
    kernel_ids = [id for id in registry.keys() if id.startswith("kernel/")]
    for kernel_id in kernel_ids:
        name = kernel_id.split("/")[-1]

        if name == "linear":
            name = "squared_euclidean"

        new_id = f"rdm/{name}"
        kernel_function = registry[kernel_id]
        derived[new_id] = kernel_to_rdm(kernel_function)


    # remove the entries in derived that are already in registry
    derived = {k: v for k, v in derived.items() if k not in registry}
    return derived


def derive_methods_recursively(registry):
    # derive until no new registry entries are derived
    derived = {}
    while True:
        new_derived = derive_methods_once(registry)
        if len(new_derived) == 0:
            break
        derived = {**derived, **new_derived}  # only the derived registry
        registry = {**registry, **new_derived}  # input registry + derived registry
        
    return derived

# derived = derive_methods_recursively(diffscore.registry)
derived = derive_methods_recursively(measures)
for k, v in derived.items():
    print("Register derived", k)
    register(k, v)
print("Number of total measures", len(measures))
# print("Number of total measures", len(diffscore.registry))


def test_measures():
    import similarity

    np.random.seed(0)
    torch.manual_seed(0)

    X = torch.rand(20, 10, dtype=torch.double)
    Y = torch.rand(20, 10, dtype=torch.double)

    # CKA Gretton
    s1 = similarity.make("measure/netrep/cka-kernel=linear-hsic=gretton-score")(np.asarray(X), np.asarray(Y))
    s2 = make("measure/cka-kernel=linear-hsic=gretton-score")(X, Y)

    Kx = make("kernel/linear")(X)
    Ky = make("kernel/linear")(Y)
    Kx_centered = make("transform/center_rows_columns")(Kx)
    Ky_centered = make("transform/center_rows_columns")(Ky)
    s3 = make("similarity/cosine")(Kx_centered, Ky_centered)

    s4 = make("similarity/centered-cosine")(Kx, Ky)

    Dx = make("transform/kernel_to_rdm")(make("kernel/linear"))(X)
    Dy = make("transform/kernel_to_rdm")(make("kernel/linear"))(Y)
    Dx = make("transform/center_rows_columns")(Dx)
    Dy = make("transform/center_rows_columns")(Dy)
    s5 = make("similarity/cosine")(Dx, Dy)

    s6 = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=cosine_cov")(np.asarray(X), np.asarray(Y))

    print("CKA gretton", s1, s2.item(), s3.item(), s4.item(), s5.item(), s6.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())
    assert np.allclose(s1, s4.item())
    assert np.allclose(s1, s5.item())
    assert np.allclose(s1, s6.item())

    
    # CKA Lange
    s1 = similarity.make("measure/netrep/cka-kernel=linear-hsic=lange-score")(np.asarray(X), np.asarray(Y))

    s2 = make("measure/cka-kernel=linear-hsic=lange-score")(X, Y)

    Dx = make("transform/kernel_to_rdm")(make("kernel/linear"))(X)
    Dy = make("transform/kernel_to_rdm")(make("kernel/linear"))(Y)
    Dx = make("transform/center_rows_columns")(Dx)
    Dy = make("transform/center_rows_columns")(Dy)
    Dx = make("transform/zero_diagonal")(Dx)
    Dy = make("transform/zero_diagonal")(Dy)
    s3 = make("similarity/cosine")(Dx, Dy)

    s4 = make("measure/rdm=squared_euclidean-similarity=(centered-zero_diagonal-cosine)-score")(X, Y)
    print("CKA lange", s1, s2.item(), s3.item(), s4.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())
    assert np.allclose(s1, s4.item())

    # NBS
    s1 = similarity.make("measure/nn_similarity_index/nbs-score")(np.asarray(X), np.asarray(Y))
    s2 = make("measure/nbs")(X, Y)

    Kx = make("kernel/linear")(X)
    Ky = make("kernel/linear")(Y)
    Kx = make("transform/center_rows_columns")(Kx)
    Ky = make("transform/center_rows_columns")(Ky)
    s3 = make("similarity/bures")(Kx, Ky)

    s4 = make("measure/kernel=linear-similarity=(centered-bures)-score")(X, Y)
    print("NBS", s1, s2.item(), s3.item(), s4.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())
    assert np.allclose(s1, s4.item())


    # Angular Procrustes
    # TODO: two different "score" versions of Procrustes
    # (i) NBS = cos(procrustes-distance=angular)
    # (ii) 1 - 2/pi * procrustes-distance=angular
    s1 = similarity.make("measure/netrep/procrustes-distance=angular")(np.asarray(X), np.asarray(Y))
    s1 = 1 - 2/np.pi * s1
    s2 = make("measure/procrustes-angular_score")(X, Y)

    s3 = make("measure/kernel=linear-similarity=(centered-bures)-angular_score")(X, Y)
    print("Angular Procrustes", s1, s2.item(), s3.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())

    # RSA euclidean, cosine
    s1 = similarity.make("measure/netrep/rsa-rdm=squared_euclidean-compare=cosine")(np.asarray(X), np.asarray(Y))
    Kx = make("kernel/linear")(X)
    Ky = make("kernel/linear")(Y)
    Dx = make("transform/kernel_to_rdm")(make("kernel/linear"))(X)
    Dy = make("transform/kernel_to_rdm")(make("kernel/linear"))(Y)
    # Dx = make("transform/zero_diagonal")(Dx)  # optional (diagonal already zero)
    # Dy = make("transform/zero_diagonal")(Dy)
    s2 = make("similarity/cosine")(Dx, Dy)  

    # Kx_zero = make("transform/zero_diagonal")(Kx)
    # Ky_zero = make("transform/zero_diagonal")(Ky)
    # s3 = make("similarity/cosine")(Kx_zero, Ky_zero)  # cos similarity of kernels != cos similarity of rdms when not centered!

    rdmX = make("rdm/squared_euclidean")(X)
    rdmY = make("rdm/squared_euclidean")(Y)
    s4 = make("similarity/cosine")(rdmX, rdmY)

    s5 = make("measure/rdm=squared_euclidean-similarity=cosine-score")(X, Y)
    print("RSA", s1, s2.item(), s4.item(), s5.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s4.item())
    assert np.allclose(s1, s5.item())

    # RSA euclidean, corr
    s1 = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=pearson")(np.asarray(X), np.asarray(Y))
    # s2 = make("measure/rdm=squared_euclidean-similarity=correlation-score")(X, Y)

    Dx = make("rdm/squared_euclidean")(X)
    Dy = make("rdm/squared_euclidean")(Y)
    def corr(R1, R2):
        # https://rsatoolbox.readthedocs.io/en/stable/comparing.html#pearson-correlation
        # get the upper triangular part as a vector
        # triu_indices = torch.triu_indices(R1.shape[0], R1.shape[1], offset=1)
        # r1 = R1[triu_indices[0], triu_indices[1]]
        # r2 = R2[triu_indices[0], triu_indices[1]]
        # r1_mean = r1.mean()
        # r2_mean = r2.mean()
        # # center the vectors
        # r1 = r1 - r1_mean
        # r2 = r2 - r2_mean
        # # # compute correlation
        # return torch.sum(r1*r2) / (torch.linalg.norm(r1) * torch.linalg.norm(r2))

        # equivalent to:
        n = R1.shape[0]
        R1 = R1 - n / (n - 1) * R1.mean()
        R2 = R2 - n / (n - 1) * R2.mean()
        return make("similarity/zero_diagonal-cosine")(R1, R2)

    s2 = corr(Dx, Dy)

    s3 = make("measure/rdm=squared_euclidean-similarity=upper_triangular_correlation-score")(X, Y)

    print("RSA euclidean, corr", s1, s2.item(), s3.item())
    assert np.allclose(s1, s2.item())

    # RSA correlation, cosine
    s1 = similarity.make("measure/rsatoolbox/rsa-rdm=correlation-compare=cosine")(np.asarray(X), np.asarray(Y))
    # ma /= np.sqrt(np.einsum('ij,ij->i', ma, ma))[:, None]
    # rdm = 1 - np.einsum('ik,jk', ma, ma)

    def corr_rdm(X):
        X_centered = X - X.mean(dim=1, keepdim=True)
        # X_centered /= torch.sqrt(torch.einsum('ij,ij->i', X_centered, X_centered))[:, None]
        # return 1 - torch.einsum('ik,jk', X_centered, X_centered)
        X_centered /= torch.sqrt(torch.sum(X_centered**2, dim=1, keepdim=True))
        return 1 - X_centered @ X_centered.T

    Cx = corr_rdm(X)
    Cy = corr_rdm(Y)
    s2 = make("similarity/cosine")(Cx, Cy)

    s3 = make("measure/rdm=correlation-similarity=cosine-score")(X, Y)
    print("RSA correlation", s1, s2.item(), s3.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())

    # RSA Bures similarity (NBS)
    s1 = similarity.make("measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=bures")(np.asarray(X), np.asarray(Y))
    rdmX = make("rdm/squared_euclidean")(X)
    rdmY = make("rdm/squared_euclidean")(Y)
    rdmX /= X.shape[1]
    rdmY /= Y.shape[1]
    # TODO: why is rsatoolbox doing -rdm/2?
    DX = -rdmX / 2
    DY = -rdmY / 2
    DX = make("transform/center_rows_columns")(DX)
    DY = make("transform/center_rows_columns")(DY)
    s2 = make("similarity/bures")(DX, DY) 

    s3 = similarity.make("similarity/rsatoolbox/zero_diagonal-bures")(rdmX, rdmY)

    # s4 = make("measure/rdm=squared_euclidean-similarity=bures-score")(X, Y)  # TODO: gives inf
    s4 = make("measure/kernel=linear-similarity=(centered-zero_diagonal-bures)-score")(X, Y)
    print("RSA bures", s1, s2.item(), s3.item(), s4.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())


    # CCA squared corr
    s1 = similarity.make("measure/contrasim/cca-squared_score")(np.asarray(X), np.asarray(Y))

    # seems that have to center before whitening
    X_centered = X - X.mean(dim=0, keepdim=True)
    Y_centered = Y - Y.mean(dim=0, keepdim=True)
    X_whitened = make("transform/whiten")(X_centered)
    Y_whitened = make("transform/whiten")(Y_centered)
    Kx = make("kernel/linear")(X_whitened)
    Ky = make("kernel/linear")(Y_whitened)
    Kx = make("transform/center_rows_columns")(Kx)
    Ky = make("transform/center_rows_columns")(Ky)
    s2 = make("similarity/cosine")(Kx, Ky)

    s3 = make("measure/kernel=(centered-whitened-linear)-similarity=cosine-score")(X, Y)
    s4 = make("measure/cca-squared_score")(X, Y)
    print("CCA squared corr", s1, s2.item(), s3.item(), s4.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())
    assert np.allclose(s1, s4.item())

    # CCA mean corr
    s1 = similarity.make("measure/contrasim/cca-score")(np.asarray(X), np.asarray(Y))
    s2 = make("similarity/bures")(Kx, Ky)
    s3 = make("measure/kernel=(centered-whitened-linear)-similarity=bures-score")(X, Y)
    s4 = make("measure/cca-score")(X, Y)
    print("CCA mean corr", s1, s2.item(), s3.item(), s4.item())
    assert np.allclose(s1, s2.item())
    assert np.allclose(s1, s3.item())
    assert np.allclose(s1, s4.item())

    # Bures distance
    bures_distance = similarity.make("measure/nn_similarity_index/bures_distance")(np.asarray(X), np.asarray(Y))
    Kx = make("kernel/linear")(X)
    Ky = make("kernel/linear")(Y)
    Kx = make("transform/center_rows_columns")(Kx)
    Ky = make("transform/center_rows_columns")(Ky)
    s = make("distance/bures")(Kx, Ky)
    print("Bures distance", bures_distance, s.item())
    assert np.allclose(bures_distance, s.item())

    # TODO: should be the same as cca (if center rdms)
    s = similarity.make("measure/rsatoolbox/rsa-rdm=mahalanobis-compare=cosine")(np.asarray(X), np.asarray(Y))
    print("RSA mahalanobis", s.item())


    # alpha procrustes
    Kx = make("kernel/linear")(X)
    Ky = make("kernel/linear")(Y)
    Kx = make("transform/center_rows_columns")(Kx)
    Ky = make("transform/center_rows_columns")(Ky)

    # TODO: why is not equal???
    s1 = make("distance/alpha_procrustes")(Kx, Ky, alpha=2)
    s2 = make("distance/euclidean")(Kx, Ky)
    print("Alpha procrustes, alpha=2", s1.item(), s2.item())

    s3 = make("distance/alpha_procrustes")(Kx, Ky, alpha=1/2)
    # 
    # alpha procrustes distance with alpha 1/2 = 2 * Bures-Wassertein distance (https://arxiv.org/pdf/1908.09275)
    s4 = 2*make("distance/bures")(Kx, Ky)
    print("Alpha procrustes, alpha=0.5", s3.item(), s4.item())
    assert np.allclose(s3.item(), s4.item())
    # assert np.allclose(s1.item(), s2.item())  # TODO

    # TODO: different alpha procrustes distances not just taking the alpha th root of the kernel matrix?
    # def matrix_sqrt(X):
    #     sX, uX = torch.linalg.eigh(X)
    #     return uX @ (torch.sqrt(torch.clip(sX[:, None], min=0.0)) * uX.T)
    # Kx_sqrt = matrix_sqrt(Kx)
    # Ky_sqrt = matrix_sqrt(Ky)
    # s5 = make("distance/alpha_procrustes")(Kx_sqrt, Ky_sqrt, alpha=2)
    # s6 = make("distance/euclidean")(Kx_sqrt, Ky_sqrt)
    # print(s5, s6)


# def reshape2d(X, Y, to_tensor=True):
#     if to_tensor:
#         X = torch.as_tensor(X)
#         Y = torch.as_tensor(Y)

#     # convert to same dtype (some measures raise error if dtype is different)
#     # TODO: raise RuntimeError: Index out of range
#     # X = X.double()
#     # Y = Y.double()

#     Y = Y.to(X.dtype)

#     if len(X.shape) == 3:
#         X = X.reshape(X.shape[0]*X.shape[1], -1)
#     if len(Y.shape) == 3:
#         Y = Y.reshape(Y.shape[0]*Y.shape[1], -1)
#     return X, Y


# class RegCCA:
#     """
#     Code adapted from https://github.com/ahwillia/netrep/blob/main/netrep/metrics/linear.py
#     """
#     def __init__(self, alpha=1, zero_pad=True, scoring_method='angular'):
#         self.alpha = alpha
#         self.zero_pad = zero_pad
#         self.scoring_method = scoring_method

#     def fit(self, X, Y):
#         X, Y = reshape2d(X, Y)
#         X = X.double()
#         Y = Y.double()
#         # zero padding
#         X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

#         # centering
#         self.mx = torch.mean(X, dim=0)
#         self.my = torch.mean(Y, dim=0)
#         X = X - self.mx
#         Y = Y - self.my

#         Xw, Zx = whiten(X, self.alpha)
#         Yw, Zy = whiten(Y, self.alpha)

#         U, _, Vt = torch.linalg.svd(Xw.T @ Yw)

#         Zx = Zx.double()
#         Zy = Zy.double()
#         self.Wx = Zx @ U
#         self.Wy = Zy @ Vt.T

#         return self

#     def score(self, X, Y):
#         X, Y = reshape2d(X, Y)
#         X = X.double()
#         Y = Y.double()
#         # zero padding
#         X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)
#         # centering
#         X = X - self.mx
#         Y = Y - self.my
#         # rotational alignment
#         X = X @ self.Wx
#         Y = Y @ self.Wy
#         # scoring
#         if self.scoring_method == 'angular':
#             normalizer = torch.linalg.norm(X.ravel()) * torch.linalg.norm(Y.ravel())
#             dist = torch.dot(X.ravel(), Y.ravel()) / normalizer
#             dist = torch.arccos(dist)
#         elif self.scoring_method == 'euclidean':
#             dist = torch.linalg.norm(X - Y, ord='fro')
#         else:
#             raise NotImplementedError
#         return dist

#     def fit_score(self, X, Y):
#         return self.fit(X, Y).score(X, Y)

#     def __call__(self, X, Y):
#         return self.fit_score(X, Y)


# class CKA:
#     def __init__(self, arccos=False):
#         self.arccos = arccos

#     def score(self, X, Y):
#         # X: time x trial x neuron
#         X, Y = reshape2d(X, Y)
#         # score = linear_CKA(X, Y)
#         # assert torch.allclose(cka_svd(X@X.T, Y@Y.T), score)
#         score = cka_svd(X@X.T, Y@Y.T)
#         return score if not self.arccos else torch.arccos(score)

#     def __call__(self, X, Y):
#         return self.score(X, Y)


# class RSA:
#     def __init__(self, arccos=False):
#         self.arccos = arccos

#     def score(self, X, Y):
#         # X: time x trial x neuron
#         X, Y = reshape2d(X, Y)

#         XX, YY = centering(X@X.T), centering(Y@Y.T)
#         score = torch.sum(XX*YY)/(torch.linalg.norm(XX.reshape(-1))*torch.linalg.norm(YY.reshape(-1)))
#         return score if not self.arccos else torch.arccos(score)

#     def __call__(self, X, Y):
#         return self.score(X, Y)



# def check_equal_shapes(X, Y, nd=2, zero_pad=False):
#     if (X.ndim != nd) or (Y.ndim != nd):
#         raise ValueError(
#             "Expected {}d arrays, but shapes were {} and "
#             "{}.".format(nd, X.shape, Y.shape)
#         )

#     if X.shape != Y.shape:

#         if zero_pad and (X.shape[:-1] == Y.shape[:-1]):

#             # Number of padded zeros to add.
#             n = max(X.shape[-1], Y.shape[-1])

#             # Padding specifications for X and Y.
#             px = torch.zeros((nd, 2), dtype=torch.int)
#             py = torch.zeros((nd, 2), dtype=torch.int)
#             # torch pad different from numpy pad!
#             px[0, -1] = n - X.shape[-1]
#             py[0, -1] = n - Y.shape[-1]

#             # Pad X and Y with zeros along final axis.
#             X = torch.nn.functional.pad(X, tuple(px.flatten()))
#             Y = torch.nn.functional.pad(Y, tuple(py.flatten()))

#         else:
#             raise ValueError(
#                 "Expected arrays with equal dimensions, "
#                 "but got arrays with shapes {} and {}."
#                 "".format(X.shape, Y.shape))

#     return X, Y


# def centering(K):
#     n = K.shape[0]
#     unit = torch.ones([n, n], dtype=K.dtype)
#     I = torch.eye(n, dtype=K.dtype)
#     H = I - unit / n

#     return (H @ K) @ H  # HKH are the same with KH, KH is the first centering, H(KH) do the second time, results are the sme with one time centering
#     # return np.dot(H, K)  # KH


# def linear_HSIC(X, Y):
#     L_X = X @ X.T
#     L_Y = Y @ Y.T
#     return torch.sum(centering(L_X) * centering(L_Y))  # <vec(XX.T, vec(YY.T)>


# def linear_CKA(X, Y):
#     hsic = linear_HSIC(X, Y)
#     var1 = torch.sqrt(linear_HSIC(X, X))
#     var2 = torch.sqrt(linear_HSIC(Y, Y))

#     return hsic / (var1 * var2)


# def cka_svd(XX, YY):
#     XX, YY = centering(XX), centering(YY)
#     # assert torch.allclose(XX, XX.T, atol=1e-5), torch.max(torch.abs(XX - XX.T))
#     # assert torch.allclose(YY, YY.T, atol=1e-5), torch.max(torch.abs(YY - YY.T))

#     lambX, uX = torch.linalg.eigh(XX)
#     lambY, uY = torch.linalg.eigh(YY)
#     uX_uY = uX.T @ uY

#     lambX_norm = torch.sqrt(torch.sum(lambX**2))
#     lambY_norm = torch.sqrt(torch.sum(lambY**2))

#     return torch.sum(torch.outer(lambX, lambY) * uX_uY * uX_uY) / (lambX_norm*lambY_norm)


# @register("measure/cka-angular")
# def cka_angular(X, Y):
#     cka = CKA(arccos=True)
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     return cka(X, Y)


# @register("measure/cka-angular-score")
# def cka_angular_score(X, Y):
#     return 1 - cka_angular(X, Y) / (np.pi/2)


# @register("measure/rsa-correlation-corr")
# def rsa_correlation_corr(X, Y):
#     rsa = RSA(arccos=False)
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     return rsa(X, Y)


# @register("measure/procrustes-angular")
# def procrustes_angular(X, Y):
#     cca = RegCCA(alpha=1)
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     return cca(X, Y)


# @register("measure/procrustes-angular-score")
# def procrustes_angular_score(X, Y):
#     return 1 - procrustes_angular(X, Y) / (np.pi/2)


# @register("measure/procrustes-angular-cv")
# def procrustes_angular_cv(X, Y, n_splits=5, fit_ratio=0.8):
#     cca = RegCCA(alpha=1)
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)

#     # cross val over conditions
#     n_conditions = X.shape[1]
#     n_fit = int(n_conditions * fit_ratio)

#     scores = torch.zeros(n_splits)
#     for i in range(n_splits):
#         indices = torch.randperm(n_conditions)
#         fit_conditions = indices[:n_fit]
#         val_conditions = indices[n_fit:]

#         fit_X = X[:, fit_conditions, :]
#         val_X = X[:, val_conditions, :]
#         fit_Y = Y[:, fit_conditions, :]
#         val_Y = Y[:, val_conditions, :]

#         cca.fit(fit_X, fit_Y)
#         scores[i] = cca.score(val_X, val_Y)
#     return torch.mean(scores)


# @register("measure/procrustes-angular-cv-score")
# def procrustes_angular_cv_score(X, Y):
#     return 1 - procrustes_angular_cv(X, Y) / (np.pi/2)


# @register("measure/procrustes-euclidean")
# def procrustes_euclidean(X, Y):
#     cca = RegCCA(alpha=1, scoring_method="euclidean")
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     return cca(X, Y)


# @register("measure/cca-angular")
# def cca_angular(X, Y):
#     cca = RegCCA(alpha=0)
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     return cca(X, Y)


# @register("measure/cca-angular-score")
# def cca_angular_score(X, Y):
#     return 1 - cca_angular(X, Y) / (np.pi/2)


# @register("measure/linreg")
# def linreg(arccos=False, zero_pad=True):
#     # X: neural data, Y: model data
#     # ref: https://arxiv.org/pdf/1905.00414.pdf
#     # R2 = 1 - min_B || X - YB ||_F^2 / || X ||_F^2 = || Q_Y.T X ||_F^2 / || X ||_F^2
#     def _fit_score(X, Y):
#         # n_steps, n_trials, n_neurons = X.shape
#         X, Y = reshape2d(X, Y)
#         # zero padding
#         X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

#         X = X - X.mean(axis=0)
#         Y = Y - Y.mean(axis=0)

#         Q, R = torch.linalg.qr(Y)
#         R2 = torch.linalg.norm(Q.T @ X) ** 2 / torch.linalg.norm(X) ** 2

#         if arccos:
#             if torch.abs(R2 - 1) < 1e-5:
#                 # arccos of 1 gives nan
#                 return torch.tensor(0.)
#             R2 = torch.arccos(R2)
#         return R2
#     return _fit_score


# register("measure/linreg-angular", partial(linreg, arccos=True))


# @register("measure/linreg-cv")
# def linreg_cv(arccos=False, zero_pad=True, n_splits=5, fit_ratio=0.8):
#     class LinRegScore:
#         def __init__(self, arccos=False, zero_pad=True):
#             self.arccos = arccos
#             self.zero_pad = zero_pad

#         def fit(self, X, Y):
#             X, Y = reshape2d(X, Y)
#             # zero padding
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

#             self.mx = X.mean(axis=0)
#             self.my = Y.mean(axis=0)
#             X = X - self.mx
#             Y = Y - self.my

#             # self.Q, self.R = torch.linalg.qr(Y)
#             # self.B = torch.linalg.solve(self.R, self.Q.T @ X)
#             self.B = torch.linalg.lstsq(Y, X).solution

#         def score(self, X, Y):
#             X, Y = reshape2d(X, Y)
#             # zero padding
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

#             X = X - self.mx
#             Y = Y - self.my

#             X_pred = Y @ self.B
#             R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
#             if self.arccos:
#                 if torch.abs(R2 - 1) < 1e-5:
#                     return torch.tensor(0.)
#                 # TODO: nan if R2 is too large
#                 R2 = torch.arccos(R2)
#             return R2

#     linreg = LinRegScore(arccos=arccos, zero_pad=zero_pad)

#     def _fit_score(X, Y):
#         # linreg.fit(X, Y)
#         # score1 = linreg.score(X, Y)
#         # score2 = make("measure.linreg", arccos=arccos)(X, Y)
#         # assert torch.allclose(score1, score2), f"{score1} != {score2}"
#         # print("Check passed")

#         # cross val over conditions
#         n_conditions = X.shape[1]
#         n_fit = int(n_conditions * fit_ratio)

#         scores = torch.zeros(n_splits)
#         for i in range(n_splits):
#             indices = torch.randperm(n_conditions)
#             fit_conditions = indices[:n_fit]
#             val_conditions = indices[n_fit:]
#             assert len(fit_conditions) + len(val_conditions) == n_conditions

#             fit_X = X[:, fit_conditions, :]
#             val_X = X[:, val_conditions, :]

#             fit_Y = Y[:, fit_conditions, :]
#             val_Y = Y[:, val_conditions, :]

#             linreg.fit(fit_X, fit_Y)
#             score = linreg.score(val_X, val_Y)
#             scores[i] = score
#         score = torch.mean(scores)
#         return score

#     return _fit_score


# register("measure/linreg-angular-cv", partial(linreg_cv, arccos=True))


# @register("measure/linreg-r2#5folds_cv")
# def measure_linreg(zero_pad=True, alpha=0, n_splits=5, agg_fun="r2"):
#     class LinRegScore:
#         def fit(self, X, Y):
#             # zero padding
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

#             self.mx = X.mean(axis=0)
#             self.my = Y.mean(axis=0)
#             X = X - self.mx
#             Y = Y - self.my

#             # fit mapping from Y to X
#             if alpha > 0:
#                 self.B = torch.linalg.lstsq(Y.T @ Y + alpha * torch.eye(Y.shape[1]), Y.T @ X).solution

#             else:
#                 # self.Q, self.R = torch.linalg.qr(Y)
#                 # self.B = torch.linalg.solve(self.R, self.Q.T @ X)
#                 self.B = torch.linalg.lstsq(Y, X).solution

#         def score(self, X, Y):
#             # zero padding
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

#             X = X - self.mx
#             Y = Y - self.my

#             X_pred = Y @ self.B

#             if agg_fun == "r2":
#                 R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
#                 return R2
#             elif agg_fun == "pearsonr":
#                 r = torch.dot(X.ravel(), X_pred.ravel()) / (torch.linalg.norm(X) * torch.linalg.norm(X_pred))
#                 # from scipy.stats import pearsonr
#                 # r_gt = pearsonr(X.ravel().detach(), X_pred.ravel().detach())
#                 # breakpoint()
#                 # print(r_gt.statistic, r)
#                 return r
#             else:
#                 raise NotImplementedError(f"agg_fun={agg_fun}")

#     linreg = LinRegScore()

#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)

#         if n_splits is None:
#             linreg.fit(X, Y)
#             score = linreg.score(X, Y)
#             return score

#         # cross val over time and conditions concatenated
#         kfold = KFold(n_splits=n_splits, shuffle=False)
#         scores = torch.zeros(n_splits)
#         for i, (train_index, test_index) in enumerate(kfold.split(X)):
#             fit_X = X[train_index]
#             val_X = X[test_index]

#             fit_Y = Y[train_index]
#             val_Y = Y[test_index]

#             linreg.fit(fit_X, fit_Y)
#             score = linreg.score(val_X, val_Y)
#             scores[i] = score
#         score = torch.mean(scores)
#         return score

#     return _fit_score


# # register(
# #     "measure.linreg-r2#no_cv",
# #     partial(measure_linreg, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda1-r2#5folds_cv",
# #     partial(measure_linreg, alpha=1)
# # )
# # register(
# #     "measure.ridge-lambda10-r2#5folds_cv",
# #     partial(measure_linreg, alpha=10)
# # )
# # register(
# #     "measure.ridge-lambda100-r2#5folds_cv",
# #     partial(measure_linreg, alpha=100)
# # )
# # register(
# #     "measure.ridge-lambda1000-r2#5folds_cv",
# #     partial(measure_linreg, alpha=1000)
# # )
# # register(
# #     "measure.ridge-lambda1-r2#no_cv",
# #     partial(measure_linreg, alpha=1, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda1-r2",
# #     partial(measure_linreg, alpha=1, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda10-r2#no_cv",
# #     partial(measure_linreg, alpha=10, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda100-r2#no_cv",
# #     partial(measure_linreg, alpha=100, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda10-r2",
# #     partial(measure_linreg, alpha=10, n_splits=None)
# # )
# # register(
# #     "measure.ridge-lambda100-r2",
# #     partial(measure_linreg, alpha=100, n_splits=None)
# # )

# # # pearsonr
# # register(
# #     "measure.linreg-pearsonr#5folds_cv",
# #     partial(measure_linreg, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.linreg-pearsonr#5folds_cv",
# #     partial(measure_linreg, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda1-pearsonr#5folds_cv",
# #     partial(measure_linreg, alpha=1, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda10-pearsonr#5folds_cv",
# #     partial(measure_linreg, alpha=10, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda100-pearsonr#5folds_cv",
# #     partial(measure_linreg, alpha=100, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda1000-pearsonr#5folds_cv",
# #     partial(measure_linreg, alpha=1000, agg_fun="pearsonr")
# # )

# # # pearsonr, no cv
# # register(
# #     "measure.linreg-pearsonr#no_cv",
# #     partial(measure_linreg, n_splits=None, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.linreg-pearsonr#no_cv",
# #     partial(measure_linreg, n_splits=None, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda1-pearsonr#no_cv",
# #     partial(measure_linreg, alpha=1, n_splits=None, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda10-pearsonr#no_cv",
# #     partial(measure_linreg, alpha=10, n_splits=None, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda100-pearsonr#no_cv",
# #     partial(measure_linreg, alpha=100, n_splits=None, agg_fun="pearsonr")
# # )
# # register(
# #     "measure.ridge-lambda1000-pearsonr#no_cv",
# #     partial(measure_linreg, alpha=1000, n_splits=None, agg_fun="pearsonr")
# # )


# for n_splits in [None, 5]:
#     for alpha in [0, 1, 10, 100, 1000, 10000]:
#         for agg_fun in ["r2", "pearsonr"]:
#             name = "linreg" if alpha == 0 else f"ridge-lambda{alpha}"
#             cv = "no_cv" if n_splits is None else f"{n_splits}folds_cv"
#             # print("registering", f"measure.{name}-{agg_fun}#{cv}")
#             register(
#                 f"measure.{name}-{agg_fun}#{cv}",
#                 partial(measure_linreg, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)
#             )

# # linear regression symmetric
# def measure_linreg_sym(zero_pad=True, alpha=0, n_splits=5, agg_fun="r2"):
#     linreg = measure_linreg(zero_pad=zero_pad, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)

#     def _fit_score(X, Y):
#         score1 = linreg(X, Y)
#         score2 = linreg(Y, X)
#         return (score1 + score2) / 2

#     return _fit_score


# for n_splits in [None, 5]:
#     for alpha in [0, 1, 10, 100, 1000, 10000]:
#         for agg_fun in ["r2", "pearsonr"]:
#             name = "linreg" if alpha == 0 else f"ridge-lambda{alpha}"
#             cv = "no_cv" if n_splits is None else f"{n_splits}folds_cv"
#             # print("registering", f"measure.{name}-{agg_fun}-sym#{cv}")
#             register(
#                 f"measure.{name}-{agg_fun}-sym#{cv}",
#                 partial(measure_linreg_sym, alpha=alpha, n_splits=n_splits, agg_fun=agg_fun)
#             )


# def kfold_crossval(measure, n_splits=5):
#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)

#         if n_splits is None:
#             measure.fit(X, Y)
#             score = measure.score(X, Y)
#             return score

#         # cross val over time and conditions concatenated
#         kfold = KFold(n_splits=n_splits, shuffle=False)
#         scores = torch.zeros(n_splits)
#         for i, (train_index, test_index) in enumerate(kfold.split(X)):
#             fit_X = X[train_index]
#             val_X = X[test_index]

#             fit_Y = Y[train_index]
#             val_Y = Y[test_index]

#             measure.fit(fit_X, fit_Y)
#             score = measure.score(val_X, val_Y)
#             scores[i] = score
#         score = torch.mean(scores)
#         return score

#     return _fit_score


# @register("measure/procrustes-angular#5folds_cv")
# def _():
#     measure = RegCCA(alpha=1)
#     return kfold_crossval(measure=measure, n_splits=5)


# @register("measure/procrustes-angular-score#5folds_cv")
# def _():
#     measure = RegCCA(alpha=1, scoring_method="euclidean")
#     def _fit_score(X, Y):
#         return 1 - measure(X, Y) / (np.pi/2)
#     return _fit_score


# @register("measure/procrustes-euclidean-score")
# def _():
#     measure = RegCCA(alpha=1)
#     def _fit_score(X, Y):
#         return 1 - kfold_crossval(measure=measure, n_splits=5)(X, Y) / (np.pi/2)
#     return _fit_score


# def test_pytorch_metric(dataset, metric_id):
#     data = dataset.get_activity()
#     cond_avg = data.mean(axis=1, keepdims=True).repeat(data.shape[1], axis=1)

#     metric = make(f"measure.{metric_id}")
#     metric2 = similarity.make(f"measure.{metric_id}")

#     score = metric(data, cond_avg)
#     score2 = metric2.fit_score(data, cond_avg)
#     assert np.allclose(score.numpy(), score2)
#     print("Test passed for", metric_id)


# @register("measure/nbs-squared")
# def _():
#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)
#         # centering
#         X = X - torch.mean(X, dim=0)
#         Y = Y - torch.mean(Y, dim=0)

#         sXY = torch.linalg.svdvals(X.T @ Y)
#         sXX = torch.linalg.svdvals(X @ X.T)
#         sYY = torch.linalg.svdvals(Y @ Y.T)

#         nbs_squared = torch.sum(sXY)**2 / (torch.sum(sXX) * torch.sum(sYY))
#         return nbs_squared
#     return _fit_score


# @register("measure/nbs")
# def _():
#     nbs_square = make("measure.nbs-squared")
#     def _fit_score(X, Y):
#         return torch.sqrt(nbs_square(X, Y))
#     return _fit_score

# @register("measure/nbs-angular")
# def _():
#     nbs = make("measure.nbs")
#     def _fit_score(X, Y):
#         return torch.acos(nbs(X, Y))
#     return _fit_score


# @register("measure/nbs-angular-score")
# def _():
#     nbs = make("measure.nbs-angular")
#     def _fit_score(X, Y):
#         return 1 - nbs(X, Y) / (np.pi/2)
#     return _fit_score

# # TODO: what about nbs-squared-angular? (don't think it is a metric like nbs-angular = procrustes-angular metric)


# @register("measure/cka")
# def _():
#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)
#         # centering
#         X = X - torch.mean(X, dim=0)
#         Y = Y - torch.mean(Y, dim=0)

#         sXY = torch.linalg.svdvals(X.T @ Y)
#         sXX = torch.linalg.svdvals(X @ X.T)
#         sYY = torch.linalg.svdvals(Y @ Y.T)

#         cka = torch.sum(sXY**2) / (torch.sqrt(torch.sum(sXX**2)) * torch.sqrt(torch.sum(sYY**2)))
#         return cka
#     return _fit_score


# @register("measure/ensd")
# def _():
#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)
#         # centering
#         X = X - torch.mean(X, dim=0)
#         Y = Y - torch.mean(Y, dim=0)

#         # https://www.biorxiv.org/content/10.1101/2023.07.27.550815v1.full.pdf
#         YtX = Y.T @ X
#         XtY = YtX.T
#         XtX = X.T @ X
#         YtY = Y.T @ Y
#         score = torch.trace(YtX @ XtY) * torch.trace(XtX) * torch.trace(YtY) / (torch.trace(XtX @ XtX) * torch.trace(YtY @ YtY))

#         return score
#     return _fit_score


# @register("measure/cka-hsic_song")
# def cka_hsic_song():
#     """
#     Code adapted from https://github.com/google-research/google-research/blob/master/representation_similarity/Demo.ipynb
#     Convert numpy to pytorch
#     """
#     def center_gram(gram):
#         n = gram.shape[0]
#         gram.fill_diagonal_(0)
#         means = torch.sum(gram, 0, dtype=torch.float64) / (n - 2)
#         means -= torch.sum(means) / (2 * (n - 1))
#         gram -= means[:, None]
#         gram -= means[None, :]
#         gram.fill_diagonal_(0)
#         return gram

#     def _fit_score(X, Y):
#         X, Y = reshape2d(X, Y)

#         gram_x = X @ X.T
#         gram_y = Y @ Y.T

#         gram_x = center_gram(gram_x)
#         gram_y = center_gram(gram_y)

#         scaled_hsic = gram_x.ravel().dot(gram_y.ravel())
#         normalization_x = torch.linalg.norm(gram_x)
#         normalization_y = torch.linalg.norm(gram_y)
#         return scaled_hsic / (normalization_x * normalization_y)

#     return _fit_score


# @register("measure/linreg-angular")
# def linreg_angular(X, Y, zero_pad=True):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     X, Y = reshape2d(X, Y)
#     # zero padding
#     X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)

#     X = X - X.mean(axis=0)
#     Y = Y - Y.mean(axis=0)

#     Q, R = torch.linalg.qr(Y)
#     R2 = torch.linalg.norm(Q.T @ X) ** 2 / torch.linalg.norm(X) ** 2

#     if torch.abs(R2 - 1) < 1e-5:
#         # arccos of 1 gives nan
#         return torch.tensor(0.)
#     return torch.arccos(R2)

# @register("measure/linreg-angular-cv")
# def linreg_angular_cv(X, Y, zero_pad=True, n_splits=5, fit_ratio=0.8):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
    
#     class LinRegScore:
#         def __init__(self, zero_pad=True):
#             self.zero_pad = zero_pad

#         def fit(self, X, Y):
#             X, Y = reshape2d(X, Y)
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

#             self.mx = X.mean(axis=0)
#             self.my = Y.mean(axis=0)
#             X = X - self.mx
#             Y = Y - self.my
#             self.B = torch.linalg.lstsq(Y, X).solution

#         def score(self, X, Y):
#             X, Y = reshape2d(X, Y)
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=self.zero_pad)

#             X = X - self.mx
#             Y = Y - self.my

#             X_pred = Y @ self.B
#             R2 = 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
#             if torch.abs(R2 - 1) < 1e-5:
#                 return torch.tensor(0.)
#             return torch.arccos(R2)

#     linreg = LinRegScore(zero_pad=zero_pad)
    
#     # cross val over conditions
#     n_conditions = X.shape[1]
#     n_fit = int(n_conditions * fit_ratio)

#     scores = torch.zeros(n_splits)
#     for i in range(n_splits):
#         indices = torch.randperm(n_conditions)
#         fit_conditions = indices[:n_fit]
#         val_conditions = indices[n_fit:]

#         fit_X = X[:, fit_conditions, :]
#         val_X = X[:, val_conditions, :]
#         fit_Y = Y[:, fit_conditions, :]
#         val_Y = Y[:, val_conditions, :]

#         linreg.fit(fit_X, fit_Y)
#         scores[i] = linreg.score(val_X, val_Y)
#     return torch.mean(scores)

# @register("measure/linreg-r2")
# def linreg_r2(X, Y, zero_pad=True, alpha=0, n_splits=5, agg_fun="r2"):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
    
#     class LinRegScore:
#         def fit(self, X, Y):
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)
#             self.mx = X.mean(axis=0)
#             self.my = Y.mean(axis=0)
#             X = X - self.mx
#             Y = Y - self.my

#             if alpha > 0:
#                 self.B = torch.linalg.lstsq(Y.T @ Y + alpha * torch.eye(Y.shape[1]), Y.T @ X).solution
#             else:
#                 self.B = torch.linalg.lstsq(Y, X).solution

#         def score(self, X, Y):
#             X, Y = check_equal_shapes(X, Y, nd=2, zero_pad=zero_pad)
#             X = X - self.mx
#             Y = Y - self.my
#             X_pred = Y @ self.B

#             if agg_fun == "r2":
#                 return 1 - torch.linalg.norm(X - X_pred) ** 2 / torch.linalg.norm(X) ** 2
#             elif agg_fun == "pearsonr":
#                 return torch.dot(X.ravel(), X_pred.ravel()) / (torch.linalg.norm(X) * torch.linalg.norm(X_pred))
#             else:
#                 raise NotImplementedError(f"agg_fun={agg_fun}")

#     linreg = LinRegScore()
#     X, Y = reshape2d(X, Y)

#     if n_splits is None:
#         linreg.fit(X, Y)
#         return linreg.score(X, Y)

#     # cross val over time and conditions concatenated
#     kfold = KFold(n_splits=n_splits, shuffle=False)
#     scores = torch.zeros(n_splits)
#     for i, (train_index, test_index) in enumerate(kfold.split(X)):
#         fit_X = X[train_index]
#         val_X = X[test_index]
#         fit_Y = Y[train_index]
#         val_Y = Y[test_index]

#         linreg.fit(fit_X, fit_Y)
#         scores[i] = linreg.score(val_X, val_Y)
#     return torch.mean(scores)

# @register("measure/nbs-squared")
# def nbs_squared(X, Y):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     X, Y = reshape2d(X, Y)
    
#     # centering
#     X = X - torch.mean(X, dim=0)
#     Y = Y - torch.mean(Y, dim=0)

#     sXY = torch.linalg.svdvals(X.T @ Y)
#     sXX = torch.linalg.svdvals(X @ X.T)
#     sYY = torch.linalg.svdvals(Y @ Y.T)

#     return torch.sum(sXY)**2 / (torch.sum(sXX) * torch.sum(sYY))

# @register("measure/nbs")
# def nbs(X, Y):
#     return torch.sqrt(nbs_squared(X, Y))

# @register("measure/nbs-angular")
# def nbs_angular(X, Y):
#     return torch.acos(nbs(X, Y))

# @register("measure/nbs-angular-score")
# def nbs_angular_score(X, Y):
#     return 1 - nbs_angular(X, Y) / (np.pi/2)

# @register("measure/cka")
# def cka(X, Y):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     X, Y = reshape2d(X, Y)
    
#     # centering
#     X = X - torch.mean(X, dim=0)
#     Y = Y - torch.mean(Y, dim=0)

#     sXY = torch.linalg.svdvals(X.T @ Y)
#     sXX = torch.linalg.svdvals(X @ X.T)
#     sYY = torch.linalg.svdvals(Y @ Y.T)

#     return torch.sum(sXY**2) / (torch.sqrt(torch.sum(sXX**2)) * torch.sqrt(torch.sum(sYY**2)))

# @register("measure/ensd")
# def ensd(X, Y):
#     X = torch.as_tensor(X)
#     Y = torch.as_tensor(Y)
#     X, Y = reshape2d(X, Y)
    
#     # centering
#     X = X - torch.mean(X, dim=0)
#     Y = Y - torch.mean(Y, dim=0)

#     YtX = Y.T @ X
#     XtY = YtX.T
#     XtX = X.T @ X
#     YtY = Y.T @ Y
#     return torch.trace(YtX @ XtY) * torch.trace(XtX) * torch.trace(YtY) / (torch.trace(XtX @ XtX) * torch.trace(YtY @ YtY))



if __name__ == "__main__":
    test_measures()
