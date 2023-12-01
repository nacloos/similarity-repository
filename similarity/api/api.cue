package api
import(
    metrics "github.com/similarity/metric"
    metric_cards "github.com/similarity/metric:card"
    "github.com/similarity/backend:backends"
)

// TODO: openapi schema?

metric: {
    for k, v in metrics {
        (k): {
            "_out_": v["_out_"]  // keep only the fields to instantiate the metric
            // TODO: use card to write dostring for the metric? (accessible with help(metric))
            // TODO: add backends id and default backend to metric cards?
            metric_cards.cards[k]
            "backends": backends.backend_by_metric[k]
            "default_backend": backends.#default_backend[k]

            // TODO?
            // "card": {"backends": ..., ...}
        }
    }
}
// TODO: rename metric to measure (more general)
"measure": metric
"backend": backends.#backends
"paper": metric_cards.papers

// TODO: metric cards in "metric" or in "card"?
// "card": "metric": metric_cards.cards

