# https://github.com/uds-lsv/xRSA-AWEs
# https://arxiv.org/pdf/2109.10179
from functools import partial
from .CKA import feature_space_linear_cka

import similarity


similarity.register(
    "measure/xrsa_awes/feature_space_linear_cka",
    feature_space_linear_cka,
)


similarity.register(
    "measure/xrsa_awes/feature_space_linear_cka-debiased_True",
    partial(feature_space_linear_cka, debiased=True),
)
