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
            // TODO: use card to write dostring for the measure? (accessible with help(measure))
            // TODO: add backends id and default backend to measure cards?
            "backends": backends.backend_by_measure[k]
            "default_backend": backends.#default_backend[k]
            // select only implemented measure
            measure.cards[k]
            // TODO: slow
            if property.measure[k] != _|_ {
                property.measure[k]
            }
        }
    }
}
// don't take measures that don't have backend implementations
// "measure": measure.cards

"backend": backends.#backends
"paper": papers

// super slow
// "backend": [string]: measure: backends.#MeasureId

// "measure": property.measure
// TODO: by default, backends: []
// TODO: some property that don't have backend implementations (cause error in try.py)
// "property": measure.property
"property": property.property
