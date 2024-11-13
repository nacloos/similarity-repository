# https://github.com/ViCCo-Group/thingsvision
# https://github.com/lciernik/similarity_consistency
from .thingsvision.core.cka import get_cka
from .thingsvision.core.rsa import compute_rdm, correlate_rdms

import similarity



def make_cka_measure(kernel, unbiased, sigma=None):
    def _measure(X, Y):
        m = X.shape[0]
        cka = get_cka(
            backend="numpy",
            m=m,
            kernel=kernel,
            unbiased=unbiased,
            sigma=sigma
        )
        return cka.compare(X, Y)
    return _measure


kernels = ["linear", "rbf"]
unbiased_values = [True, False]
sigma = 1.0  # TODO

for kernel in kernels:
    for unbiased in unbiased_values:
        # name code taken from https://github.com/lciernik/similarity_consistency/blob/c7972862a38ac97990ca82399613d2c9b15dbefb/sim_consistency/tasks/model_similarity.py#L8
        method_name = f"cka_kernel_{kernel}{'_unbiased' if unbiased else '_biased'}"
        if kernel == 'rbf':
            method_name += f"_sigma_{sigma}"

        similarity.register(
            f"measure/thingsvision/{method_name}",
            make_cka_measure(kernel=kernel, unbiased=unbiased, sigma=sigma),
            function=True
        )


def make_rsa_measure(rsa_method, corr_method):
    def _measure(X, Y):
        rdm_X = compute_rdm(X, method=rsa_method)
        rdm_Y = compute_rdm(Y, method=rsa_method)
        return correlate_rdms(rdm_X, rdm_Y, correlation=corr_method)
    return _measure


rsa_methods = ["correlation", "cosine", "euclidean", "gaussian"]
corr_methods = ["pearson", "spearman"]  # TODO: can be any function in scipy.stats

for rsa_method in rsa_methods:
    for corr_method in corr_methods:
        # https://github.com/lciernik/similarity_consistency/blob/c7972862a38ac97990ca82399613d2c9b15dbefb/sim_consistency/tasks/model_similarity.py#L8
        name = f"rsa_method_{rsa_method}_corr_method_{corr_method}"

        similarity.register(
            f"measure/thingsvision/{name}",
            make_rsa_measure(rsa_method=rsa_method, corr_method=corr_method),
            function=True
        )
