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

    similarity.register(
        f"measure/resi/{name}",
        # assume inputs of format n x d
        partial(measure, shape="nd"),
        function=True
    )