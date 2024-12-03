from functools import partial
import similarity

from .metrics import AlignmentMetrics
from .metrics import hsic_biased, hsic_unbiased


register = partial(
    similarity.register,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

register(
    "platonic/cka",
    AlignmentMetrics.cka,
)
register(
    "platonic/unbiased_cka",
    AlignmentMetrics.unbiased_cka,
)

similarity.register("hsic/platonic/gretton", hsic_biased)
similarity.register("hsic/platonic/song", hsic_unbiased)


def cka(X, Y, sigma=1.0, **kwargs):
    # rename parameter 'rbf_sigma' to 'sigma'
    return AlignmentMetrics.cka(X, Y, rbf_sigma=sigma, **kwargs)

# TODO: parameters
# TODO: normalize features first?
# similarity.register(
#     "platonic/cka_rbf",
#     # rename parameter 'rbf_sigma' to 'sigma'
#     partial(AlignmentMetrics.cka, kernel_metric='rbf', rbf_sigma=1.0),
#     preprocessing=["array_to_tensor"],
#     postprocessing=["tensor_to_float"]
# )
# similarity.register(
#     "platonic/unbiased_cka_rbf",
#     partial(AlignmentMetrics.unbiased_cka, kernel_metric='rbf', rbf_sigma=1.0),
#     preprocessing=["array_to_tensor"],
#     postprocessing=["tensor_to_float"]
# )

register(
    "platonic/cka_rbf",
    partial(cka, kernel_metric='rbf', sigma=1.0),
)
register(
    "platonic/unbiased_cka_rbf",
    partial(cka, kernel_metric='rbf', sigma=1.0, unbiased=True),    
)
register(
    "platonic/cycle_knn_topk",
    partial(AlignmentMetrics.cycle_knn, topk=10),
)
register(
    "platonic/mutual_knn_topk",
    partial(AlignmentMetrics.mutual_knn, topk=10),
)
register(
    "platonic/lcs_knn_topk",
    partial(AlignmentMetrics.lcs_knn, topk=10),
)
register(
    "platonic/cknna_topk",
    partial(AlignmentMetrics.cknna, topk=10),
)
register(
    "platonic/edit_distance_knn_topk",
    partial(AlignmentMetrics.edit_distance_knn, topk=10),
)

def svcca(X, Y, dim=10, **kwargs):
    # rename 'cca_dim' to 'dim'
    return AlignmentMetrics.svcca(X, Y, cca_dim=dim, **kwargs)

register(
    "platonic/svcca",
    svcca,
)
