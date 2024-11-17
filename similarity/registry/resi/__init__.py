# https://github.com/mklabunde/resi
from functools import partial
from pathlib import Path
import sys


import similarity


# renamed package to 'resi' because of conflicting package name 'repsim'
dir_path = Path(__file__).parent
sys.path.append(str(dir_path))
from .resi.measures import ALL_MEASURES

for name, measure in ALL_MEASURES.items():
    if name == "GeometryScore":
        # skip because of dependency installation issues
        continue
    if name == "RSA":
        # RSA registered separately below
        continue

    similarity.register(
        f"resi/{name}",
        # assume inputs of format n x d
        partial(measure, shape="nd"),
    )


from resi.measures.rsa import representational_similarity_analysis

for inner in ["correlation", "euclidean"]:
    for outer in ["euclidean", "spearman"]:
        similarity.register(
            f"resi/RSA_{inner}_{outer}",
            partial(representational_similarity_analysis, shape="nd", inner=inner, outer=outer),
        )
