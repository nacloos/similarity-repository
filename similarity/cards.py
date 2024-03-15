from similarity import register, make


score_measures = [
    "cka",
    "cka-angular-score",
    "procrustes-angular-score",
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


# default cards
cards = {}
measures = make("measure.*.*")
for measure_id in measures.keys():
    cards[measure_id.split(".")[-1]] = {
        "props": []
    }

# add score prop to cards
for measure_id in score_measures:
    cards[measure_id]["props"].append("score")

# add metric prop to cards
for measure_id in distance_metrics:
    cards[measure_id]["props"].append("metric")

# register cards
for measure_id, card in cards.items():
    register(f"card.{measure_id}", card)
