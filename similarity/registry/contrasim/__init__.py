# https://github.com/technion-cs-nlp/ContraSim
# https://arxiv.org/pdf/2303.16992
from functools import partial
from .contrasim import cka, pwcca, cca_core

import similarity


similarity.register(
    "contrasim/feature_space_linear_cka",
    cka.feature_space_linear_cka,
    function=True
)


similarity.register(
    "contrasim/feature_space_linear_cka_debiased",
    partial(cka.feature_space_linear_cka, debiased=True),
    function=True
)


similarity.register(
    "contrasim/pwcca",
    pwcca.compute_pwcca,
    function=True
)


similarity.register(
    "contrasim/cca",
    cca_core.compute_cca,
    function=True
)


similarity.register(
    "contrasim/svcca",
    cca_core.compute_svcca,
    function=True
)