"""
Cards contain metadata about similarity measures.
Reference: [(Klabunde et al., 2023)](https://arxiv.org/pdf/2305.06329.pdf)
"""
from collections import defaultdict
from similarity import register, make


cards = {
    "euclidean": {
        "name": "Euclidean"
    },
    "permutation": {
        "name": "Permutation"
    },
    "correlation": {
        "name": "Correlation"
    },
    "cca": {
        "name": "Mean Canonical Correlation",
        "paper": ["yanai1974", "raghu2017", "kornblith2019"]
    },
    "cca-angular": {
        "name": "CCA Angular",
    },
    "cca-angular-score": {
        "name": "CCA Angular Score"
    },
    "cca_mean_sq_corr": {
        "name": "Mean Squared Canonical Correlation"
    },
    "svcca": {
        "name": "Singular Vector Canonical Correlation Analysis",
        "paper": "raghu2017"
    },
    "pwcca": {
        "name": "Projection-Weighted Canonical Correlation Analysis",
        "paper": "morcos2018"
    },
    "riemannian_metric": {
        "name": "Riemannian Metric",
        "paper": "shahbazi2021"
    },
    "procrustes": {
        "name": "Orthogonal Procrustes",
        "paper": ["ding2021", "williams2021"]
    },
    "procrustes-euclidean": {
        "name": "Procrustes Euclidean"
    },
    "procrustes-angular": {
        "name": "Procrustes Angular"
    },
    "procrustes-sq-euclidean": {
        "name": "Procrustes Squared Euclidean"
    },
    "procrustes-angular-score": {
        "name": "Procrustes Angular Score"
    },
    "shape_metric-angular": {
        "name": "Angular Shape Metric",
        "paper": "williams2021"
    },
    "shape_metric-euclidean": {
        "name": "Euclidean Shape Metric",
        "paper": "williams2021"
    },
    "pls": {
        "name": "Partial Least Squares"
    },
    "linear_regression": {
        "name": "Linear Regression",
        "paper": ["li2016", "kornblith2019"]
    },
    "aligned_cosine": {
        "name": "Aligned Cosine Similarity",
        "paper": "hamilton2016a"
    },
    "corr_match": {
        "name": "Correlation Match",
        "paper": "li2016"
    },
    "max_match": {
        "name": "Maximum Match",
        "paper": "wang2018"
    },
    "rsm_norm": {
        "name": "Representational Similarity Matrix Norms",
        "paper": ["shahbazi2021", "yin2018"]
    },
    "rsa": {
        "name": "Representational Similarity Analysis",
        "paper": "kriegeskorte2008"
    },
    "cka": {
        "name": "Centered Kernel Alignment",
        "paper": "kornblith2019"
    },
    "cka-angular": {
        "name": "CKA Angular",
        "paper": ["williams2021", "lange2022"]
    },
    "cka-angular-score": {
        "name": "CKA Angular Score"
    },
    "dcor": {
        "name": "Distance Correlation",
        "paper": "szekely2007"
    },
    "nbs": {
        "name": "Normalized Bures Similarity",
        "paper": "tang2020"
    },
    "nbs-angular-score": {
        "name": "NBS Angular Score"
    },
    "nbs-squared": {
        "name": "NBS Squared"
    },
    "bures_distance": {
        "name": "Bures Distance",
        "paper": "bhatia2017"
    },
    "eos": {
        "name": "Eigenspace Overlap Score",
        "paper": "may2019"
    },
    "gulp": {
        "name": "Unified Linear Probing",
        "paper": "boixadsera2022"
    },
    "riemmanian_metric": {
        "name": "Riemmanian Distance",
        "paper": "shahbazi2021"
    },
    "jaccard": {
        "name": "Jaccard",
        "paper": ["schumacher2021", "wang2020", "hryniowski2020", "gwilliam2022"]
    },
    "second_order_cosine": {
        "name": "Second-Order Cosine Similarity",
        "paper": "hamilton2016b"
    },
    "rank_similarity": {
        "name": "Rank Similarity",
        "paper": "wang2020"
    },
    "joint_rank_jaccard": {
        "name": "Joint Rank and Jaccard Similarity",
        "paper": "wang2020"
    },
    "gs": {
        "name": "Geometry Score",
        "paper": "khrulkov2018"
    },
    "imd": {
        "name": "Multi-scale Intrinsic Distance",
        "paper": "tsitsulin2020"
    },
    "rtd": {
        "name": "Representation Topology Divergence",
        "paper": "barannikov2022"
    },
    "intrinsic_dimension": {
        "name": "Intrinsic Dimension",
        "paper": "camastra2016"
    },
    "magnitude": {
        "name": "Magnitude",
        "paper": "wang2020"
    },
    "concentricity": {
        "name": "Concentricity",
        "paper": "wang2020"
    },
    "uniformity": {
        "name": "Uniformity",
        "paper": "wang2022"
    },
    "tolerance": {
        "name": "Tolerance",
        "paper": "wang2021"
    },
    "knn_graph_modularity": {
        "name": "kNN-Graph Modularity",
        "paper": "lu2022"
    },
    "neuron_graph_modularity": {
        "name": "Neuron-Graph Modularity",
        "paper": "lange2022"
    }
}


# measures where higher score is better and perfect similarity is 1
score_measures = [
    "cka",
    "cka-angular-score",
    "procrustes-angular-score",
    "cca",
    "cca-angular-score",
    "nbs",
    "nbs-angular-score",
    "nbs-squared",
    "rsa-correlation-corr"
]

# measures that satisfy the axioms of a distance metric
distance_metrics = [
    "procrustes-angular",
    "cca-angular",
]

# following Table 1 in (Klabunde et al., 2023) - might not be satisfied by the specific implementations!
# more work is needed to verify these invariances for each measure
invariances = {
    "cca": ["permutation", "orthogonal", "isotropic-scaling", "invertible-linear", "translation", "affine"],
    "rsa": ["permutation", "isotropic-scaling", "translation"],
    "cka": ["permutation", "orthogonal", "isotropic-scaling", "translation"],
    "nbs": ["permutation", "orthogonal", "isotropic-scaling"],
    "procrustes-angular": ["permutation", "orthogonal", "isotropic-scaling"],
}
invariances["procrustes-angular-score"] = invariances["procrustes-angular"]
invariances["nbs-angular-score"] = invariances["nbs"]
invariances["nbs-squared"] = invariances["nbs"]
invariances["cca-angular-score"] = invariances["cca"]
invariances["cka-angular-score"] = invariances["cka"]
invariances["rsa-correlation-corr"] = invariances["rsa"]


names = {k: v["name"] for k, v in cards.items()}
papers = {k: v.get("paper", []) for k, v in cards.items()}


def make_card(measure_id):
    props = []
    if measure_id in score_measures:
        props.append("score")
    if measure_id in distance_metrics:
        props.append("metric")
    return {
        "props": props,  # TODO: deprecated, backward compatibility
        "score": measure_id in score_measures,
        "metric": measure_id in distance_metrics,
        "name": names.get(measure_id, measure_id),
        "invariances": invariances.get(measure_id, []),
        "paper": papers.get(measure_id, [])
    }


measures = make("measure.*.*")
all_measures = [
    *measures.keys(),
    *score_measures,
    *distance_metrics
]

# create and register cards
for full_id in all_measures:
    measure_id = full_id.split(".")[-1]
    card = make_card(measure_id)
    # card.measure.{measure_id} only?
    register(f"card.measure.{measure_id}", card)
    # backward compatibility
    register(f"card.{measure_id}", card)
