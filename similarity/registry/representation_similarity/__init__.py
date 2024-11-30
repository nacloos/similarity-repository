"""
https://github.com/google-research/google-research/blob/master/representation_similarity
"""
from functools import partial
import numpy as np

from .demo import *

import similarity

similarity.register(
    "paper/representation_similarity",
    {
        "id": "kornblith2019",
        "github": "https://github.com/google-research/google-research/blob/master/representation_similarity"
    }
)

# TODO: register representation_similarity/gram_linear and use standardization
similarity.register("kernel/representation_similarity/linear", gram_linear)
similarity.register("kernel/representation_similarity/rbf-threshold={threshold}", gram_rbf)
similarity.register("similarity/representation_similarity/centered-cosine", cka)

similarity.register("representation_similarity/cca", cca, preprocessing=["center_columns"])
similarity.register("representation_similarity/cka", feature_space_linear_cka)
similarity.register("representation_similarity/cka_debiased", partial(feature_space_linear_cka, debiased=True))



# register = partial(
#   similarity.register,
#   function=True,
#   preprocessing=[
#     "reshape2d",
#     "center_columns"
#   ]
# )



# # register measures (not part of the original code)
# # TODO: different ids for feature space and gram matrix cka? (one is faster if num_features < num_examples)
# register("measure/representation_similarity/cka-hsic_gretton", feature_space_linear_cka)
# register("measure/representation_similarity/cka-hsic_song", partial(feature_space_linear_cka, debiased=True))
# register("measure/representation_similarity/cca", cca)
# # TODO: nonlinear cka
# # register("measure/representation_similarity/cka-hsic_gretton-rbf")

