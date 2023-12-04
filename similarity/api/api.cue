package api
import(
    "github.com/similarity/measure"
    "github.com/similarity/measure:property"
    "github.com/similarity/backend:backends"
    "github.com/similarity/papers"
)

"measure": {
    for k, v in backends.measures {
        (k): {
            "_out_": v["_out_"]  // keep only the fields to instantiate the measure
            "backends": backends.backend_by_measure[k]
            "default_backend": backends.#default_backend[k]
            // select only implemented measure
            measure.cards[k]
            if property.measure[k] != _|_ {
                property.measure[k]
            }
        }
    }
}

"backend": backends.#backends
"paper": papers
"property": property.property

// slow
// "backend": [string]: measure: backends.#MeasureId

