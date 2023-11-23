// can't use the backend package because would cause circular import
package backends
import(
    // "github.com/similarity"
    "github.com/similarity/backend"
    netrep "github.com/similarity/backend/alignment/netrep:backend"
    brainscore "github.com/similarity/backend/brainscore:backend"
    yuanli2333 "github.com/similarity/backend/similarity_matrix/yuanli2333:backend"
    rsatoolbox "github.com/similarity/backend/similarity_matrix/rsatoolbox:backend"
)
// TODO: everytime you want to make one metric it will validate all the backends
// not very efficient...
#backends: {
    // will validate the backends
    "netrep": netrep
    "brainscore": brainscore
    "yuanli2333": yuanli2333
    "rsatoolbox": rsatoolbox
}
// default backend choice for each metric
#default_backend: {
    procrustes: "netrep"
    cca: "netrep"
    svcca: "netrep"
    cka: "yuanli2333"
    rsa: "rsatoolbox"
    pls: "brainscore"
}
// TODO: extract names from type #MetricName?
// used in metric.cue to create the metric fields

// id instead of name? e.g. _default_backend: [#MetricId]: #BackendId  // TODO?
#default_backend: [#MetricName]: #BackendName

#metric_names: backend.#metric_names
#MetricName: backend.#MetricName
// #MetricName: similarity.#MetricName
// #BackendName: similarity.#BackendName
#BackendName: or([for id, _ in #backends { id }])

cards: {
    for id, backend in #backends {
        (id): backend.card 
    }
}
