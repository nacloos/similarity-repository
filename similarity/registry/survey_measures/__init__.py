"""
Code adapted from https://github.com/mklabunde/survey_measures/blob/main/appendix_procrustes.ipynb
"""
import numpy as np
from scipy.linalg import orthogonal_procrustes

import similarity

similarity.register(
    "measure/mklabunde",
    {
        "paper_id": "klabunde2023",
        "github": "https://github.com/mklabunde/survey_measures"
    }
)


@similarity.register(
    "survey_measures/procrustes",
    preprocessing=[
        "center_columns",
        {"id": "zero_padding", "inputs": ["X", "Y"]}
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