# https://github.com/IlyaTrofimov/RTD
from .rtd import cka, cca_core, pwcca, svcca

import similarity


similarity.register("rtd/cka", cka.cka, preprocessing=["center_columns"])

def pwcca_measure(X, Y):
    # original function returns a tuple
    return pwcca.compute_pwcca(X, Y)[0]

# TODO: numpy.linalg.LinAlgError: SVD did not converge
similarity.register(
    "rtd/pwcca",
    pwcca_measure,
    preprocessing=["transpose"]
)

similarity.register(
    "rtd/svcca",
    svcca.svcca,
    preprocessing=["transpose"]
)


similarity.register("hsic/rtd/gretton", cka.hsic)
