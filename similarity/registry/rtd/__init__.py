# https://github.com/IlyaTrofimov/RTD
from .rtd import cka, cca_core, pwcca, svcca

import similarity


similarity.register(
    "measure/rtd/cka",
    cka.cka,
    preprocessing=[
        "center_columns"
    ]
)

def pwcca_measure(X, Y):
    # original function returns a tuple
    return pwcca.compute_pwcca(X, Y)[0]

similarity.register(
    "measure/rtd/pwcca",
    pwcca_measure,
    preprocessing=[
        "transpose",
        "center_columns"
    ]
)

similarity.register(
    "measure/rtd/svcca",
    svcca.svcca,
    preprocessing=[
        "transpose",
        "center_columns"
    ]
)


similarity.register("hsic/rtd/gretton", cka.hsic)
