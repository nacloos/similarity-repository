from functools import partial
import similarity

import repsim


similarity.register(
    "measure/repsim",
    {
        "paper_id": "lange2023",
        "github": "https://github.com/wrongu/repsim/"
    }
)


register = partial(
    similarity.register,
    function=True,
    preprocessing=[
        "reshape2d",
        "array_to_tensor",
    ],
    postprocessing=[
        "tensor_to_float",
    ]
)

register(
    "measure/repsim/cka-hsic_lange-angular",
    partial(repsim.compare, method="angular_cka")
)
register(
    "measure/repsim/procrustes-angular",
    partial(
        repsim.compare,
        method="angular_shape_metric",
        alpha=1,
        # vary this param?
        # nb of components to keep (value used in the paper)
        p=100
    )
)
register(
    "measure/repsim/procrustes-euclidean",
    partial(
        repsim.compare,
        method="euclidean_shape_metric",
        alpha=1,
        p=100
    )
)
register(
    "measure/repsim/cca-angular",
    partial(
        repsim.compare,
        method="angular_shape_metric",
        alpha=0,
        p=100
    )
)
register(
    "measure/repsim/cca-euclidean",
    partial(
        repsim.compare,
        method="euclidean_shape_metric",
        alpha=0,
        p=100
    )
)
register(
    "measure/repsim/riemannian_metric",
    partial(
        repsim.compare,
        method="affine_invariant_riemannian"
    )
)