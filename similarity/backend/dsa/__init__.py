import similarity
import DSA


similarity.register(
    "measure.dsa",
    {
        "paper_id": "ostrow2023",
        "github": "https://github.com/mitchellostrow/DSA"
    }
)

# TODO: convert to standard interface (DSA class takes X, Y in constructor instead of fit_score)
# similarity.register(
#     "measure.dsa.dsa",
#     lambda: DSA.DSA
# )
