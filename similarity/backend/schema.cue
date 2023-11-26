package backend
import(
    // can't import metric here because it will cause a circular dependency
    // => move schemas to similarity package...
    "github.com/similarity"
    "github.com/similarity/utils"
    "github.com/similarity/processing"
    // TODO: if user different card package in metric, then cannot just do similarity.make(package="metric", ...)
    // metric_cards "github.com/similarity/metric:card"
)
#target: utils.#target
#Metric: similarity.#Metric
// #MetricName: similarity.#MetricName

// TODO: get metric names from cards (but can't import metric here because of cyclic import)
#metric_names: [
    "procrustes", 
    "cca", 
    "cca_mean_sq_corr",
    "pwcca",
    "svcca", 
    "cka", 
    "rsa", 
    "linear_regression",
    "pls"
]
// #metric_names: [for k, _ in metric_cards.cards { k }]
#MetricName: or(#metric_names)

#Card: {
    id?: string,  // TODO: automatically extract id?
    name?: string
    github?: string
    website?: string
    citation?: string | [...string]
}
card: #Card


// close restrict the keys of metric to be in #MetricName
metric: close({
    // #Metric doesn't do anything by default, it just adds functionalities
    // [#MetricName]: #Metric  // why is it so slow???
    [#MetricName]: { ... }
})


// for k, v in metric {
//     metric: (k): #Metric & v
// }

// prevode helper functions for pre and post processing
#reshape2d: processing.#reshape2d
#arccos: processing.#arccos
#angular_dist_to_score: processing.#angular_dist_to_score
#pca: processing.#pca

// #MetricBackend: {
//     [similarity.#MetricName]: { ... }
// }

// metric: #MetricBackend

// TODO: enforce metric names
// metric: [similarity.#MetricName]: { ... }

// metric: [metric_name=string]: {
//     #assert: metric_name & similarity.#MetricName
//     ...
// }

// metric: {
//     hello: {...}
// }


// x: close({a: 10})
// x: b: 1


// TODO: not raise any error...
// metric: close({
//     [name=("cca" | "cka")]: { ... }
// })

