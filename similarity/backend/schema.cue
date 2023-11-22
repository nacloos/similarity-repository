package backend
import(
    // can't import metric here because it will cause a circular dependency
    // => move schemas to similarity package...
    "github.com/similarity"
    "github.com/similarity/processing"
)

#Metric: similarity.#Metric
#MetricName: similarity.#MetricName

// close restrict the keys of metric to be in #MetricName
metric: close({
    [#MetricName]: {...}
})

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

