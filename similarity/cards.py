from collections import defaultdict
from similarity import register, make


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

distance_metrics = [
    "procrustes-angular",
    "cca-angular",
]

names = {
    "cka": "Centered Kernel Alignment",
    "cka-angular-score": "CKA Angular Score",
    "procrustes-angular-score": "Procrustes Angular Score",
    "nbs": "Normalized Bures Similarity",
    "nbs-angular-score": "NBS Angular Score",
    "nbs-squared": "NBS Squared",
    "cca": "Canonical Correlation Analysis",
    "cca-angular-score": "CCA Angular Score",
    "rsa-correlation-corr": "Representational Similarity Analysis",
}

invariances = {
    "cka": ["orthogonal", "isotropic-scaling"],
    "nbs": ["orthogonal", "isotropic-scaling"],
    "cca": ["invertible-linear"],
}


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
    # TODO: card.measure.{measure_id} instead?
    register(f"card.{measure_id}", card)
