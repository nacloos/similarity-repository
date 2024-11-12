from functools import partial
import similarity

from .metrics import AlignmentMetrics


similarity.register(
    "measure.platonic",
    {
        "paper_id": "huh2024prh",
        "github": "https://github.com/minyoungg/platonic-rep"
    }
)

register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d",
        "array_to_tensor"
    ]
)

register(
    "measure.platonic.cka",
    AlignmentMetrics.cka
)
# TODO: normalize features first?
register(
    "measure.platonic.cka-rbf1",
    partial(AlignmentMetrics.cka, kernel_metric='rbf', rbf_sigma=1.0)
)
register(
    "measure.platonic.cka-hsic_song",
    AlignmentMetrics.unbiased_cka
)
register(
    "measure.platonic.cycle_knn-topk10",
    partial(AlignmentMetrics.cycle_knn, topk=10)
)
register(
    "measure.platonic.mutual_knn-topk10",
    partial(AlignmentMetrics.mutual_knn, topk=10)
)
register(
    "measure.platonic.lcs_knn-topk10",
    partial(AlignmentMetrics.lcs_knn, topk=10)
)
register(
    "measure.platonic.cknna-topk10",
    partial(AlignmentMetrics.cknna, topk=10)
)
register(
    "measure.platonic.svcca-dim10",
    partial(AlignmentMetrics.svcca, cca_dim=10)
)
register(
    "measure.platonic.edit_distance_knn-topk10",
    partial(AlignmentMetrics.edit_distance_knn, topk=10)
)


