package api
import(
    // measures "github.com/similarity/measure"
    measure_cards "github.com/similarity/measure:card"
    "github.com/similarity/measure:property"
    "github.com/similarity/backend:backends"
)

// TODO: openapi schema?

measure: {
    // for k, v in measures {
    for k, v in backends.measures {
        (k): {
            "_out_": v["_out_"]  // keep only the fields to instantiate the measure
            // TODO: use card to write dostring for the measure? (accessible with help(measure))
            // TODO: add backends id and default backend to measure cards?
            measure_cards.cards[k]
            "backends": backends.backend_by_measure[k]
            "default_backend": backends.#default_backend[k]

            // TODO?
            // "card": {"backends": ..., ...}
        }
    }
}
// TODO: rename measure to measure (more general)
"measure": measure

"backend": backends.#backends
"paper": measure_cards.papers

// super slow
// "backend": [string]: measure: backends.#MeasureName

"measure": property.measure
"property": measure_cards.property

// TODO: measure cards in "measure" or in "card"?
// "card": "measure": measure_cards.cards

