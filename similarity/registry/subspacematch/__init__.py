from functools import partial
import similarity

from . import score


similarity.register(
    "measure/subspacematch",
    {
        "paper_id": "wang2018",
        "github": "https://github.com/MeckyWu/subspace-match"
    }
)

similarity.register(
    "measure/subspacematch/max_match",
    # vary epsilon?
    partial(score.maximum_match_score, epsilon=0.25),
    preprocessing=[
        "reshape2d",
        "transpose"
    ]
)