from functools import partial
import similarity

from .metrics import AlignmentMetrics
from .metrics import hsic_biased, hsic_unbiased


similarity.register(
    "paper/platonic",
    {
        "id": "huh2024prh",
        "github": "https://github.com/minyoungg/platonic-rep"
    }
)

similarity.register(
    "platonic/cka",
    AlignmentMetrics.cka,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/unbiased_cka",
    AlignmentMetrics.unbiased_cka,
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register("hsic/platonic/gretton", hsic_biased)
similarity.register("hsic/platonic/song", hsic_unbiased)


# TODO: parameters
# TODO: normalize features first?
similarity.register(
    "platonic/cka_rbf",
    partial(AlignmentMetrics.cka, kernel_metric='rbf', rbf_sigma=1.0),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/unbiased_cka_rbf",
    partial(AlignmentMetrics.unbiased_cka, kernel_metric='rbf', rbf_sigma=1.0),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)

similarity.register(
    "platonic/cycle_knn_topk",
    partial(AlignmentMetrics.cycle_knn, topk=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/mutual_knn_topk",
    partial(AlignmentMetrics.mutual_knn, topk=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/lcs_knn_topk",
    partial(AlignmentMetrics.lcs_knn, topk=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/cknna_topk",
    partial(AlignmentMetrics.cknna, topk=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/svcca",
    partial(AlignmentMetrics.svcca, cca_dim=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
similarity.register(
    "platonic/edit_distance_knn_topk",
    partial(AlignmentMetrics.edit_distance_knn, topk=10),
    preprocessing=["array_to_tensor"],
    postprocessing=["tensor_to_float"]
)
