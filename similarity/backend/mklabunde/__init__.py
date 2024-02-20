"""
Code adapted from https://github.com/mklabunde/survey_measures/blob/main/appendix_procrustes.ipynb
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes

import similarity

similarity.register(
    "measure.mklabunde",
    {
        "paper_id": "klabunde2023",
        "github": "https://github.com/mklabunde/survey_measures"
    }
)


@similarity.register(
    "measure.mklabunde.procrustes-sq-euclidean",
    function=True,
    preprocessing=[
        "reshape2d",
        "center_columns",
        # TODO: zero_padding takes two args
        # "zero_padding"
    ]
)
def procrustes(X, Y):
    r, scale = orthogonal_procrustes(X, Y)
    total_norm = (
        -2 * scale
        + np.linalg.norm(X, ord="fro") ** 2
        + np.linalg.norm(Y, ord="fro") ** 2
    )
    return total_norm
