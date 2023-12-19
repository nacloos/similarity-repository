package api
import(
    // "list"
    measure_pkg "github.com/similarity/measure"
    // "github.com/similarity/measure:property"
    "github.com/similarity/backend:backends"
    "github.com/similarity/papers"
)

measure: {
    for k, v in backends.measures {
        (k): {
            "_out_": v["_out_"]  // keep only the fields to instantiate the measure
            "backends": backends.backend_by_measure[k]
            "default_backend": backends.#default_backend[k]
            // select only implemented measure
            measure_pkg.cards[k]
            // if property.measure[k] != _|_ {
            //     property.measure[k]
            // }
        }
    }
}
// TODO: add measure_id to each measure so that know which measure is used by default?
// select a typical set of metrics among all the measures
metric: {
    permutation:    measure["permutation-angular"]
    procrustes:     measure["procrustes-angular"]
    cka:            measure["procrustes-angular"]
    cca:            measure["cca-angular"]
}

// select a typical set of score measures among all the measures
score: {
    // permutation: measure["permutation-score"]
    // procrustes: measure["procrustes-score"]
    // cka: measure["procrustes-score"]
    // cca: measure["cca-score"]
    svcca:          measure["svcca-var95"]
    pwcca:          measure["pwcca"]
    // pls
    rsa:            measure["rsa-euclidean-cosine"]
    cka:            measure["cka"]
}

// metric: {
//     // all the measures that have the "metric" property (i.e. measure distances)
//     for k, v in measure
//     if list.Contains(v["properties"], "metric") {
//         (k): v
//     }
// }
// score: {
//     for k, v in measure
//     if list.Contains(v["properties"], "score") {
//         (k): v
//     }
// }

backend: backends.#backends
paper: papers
property: measure_pkg.property

// slow
// "backend": [string]: measure: backends.#MeasureId

