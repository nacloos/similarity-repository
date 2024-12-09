# https://github.com/qihongl/nnsrm-neurips18
# https://arxiv.org/pdf/1811.11684
import numpy as np
from .qmvpa.rsa import within_RSMs, correlate_2RSMs, inter_procrustes, isc

import similarity


def rsa_measure(X, Y):
    rsms = within_RSMs([X, Y])
    return correlate_2RSMs(rsms[0], rsms[1])[0]


similarity.register(
    "nnsrm_neurips18/rsa",
    rsa_measure,
    preprocessing=["transpose"]
)


def procrustes_measure(X, Y):
    return inter_procrustes(np.array([X, Y]))[1, 0]

# TODO
similarity.register(
    "nnsrm_neurips18/procrustes",
    procrustes_measure,
    preprocessing=["transpose"]
)


def isc_measure(X, Y):
    return np.mean(isc(X, Y)[0])

# TODO: what is isc? just correlation?
similarity.register(
    "nnsrm_neurips18/isc",
    isc_measure,
    preprocessing=["transpose"]
)