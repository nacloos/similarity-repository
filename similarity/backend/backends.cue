// can't use the backend package because would cause circular import
package backends
import(
    "list"
    // "github.com/similarity"
    "github.com/similarity/backend"
    netrep      "github.com/similarity/backend/netrep:backend"
    brainscore  "github.com/similarity/backend/brainscore:backend"
    yuanli2333  "github.com/similarity/backend/yuanli2333:backend"
    rsatoolbox  "github.com/similarity/backend/rsatoolbox:backend"
    scipy       "github.com/similarity/backend/scipy:backend"
    sim_metric  "github.com/similarity/backend/sim_metric:backend"
    svcca       "github.com/similarity/backend/svcca:backend"
)

#backends: [string]: _  // schema
#backends: {
    // will validate the backends
    "netrep": netrep
    "brainscore": brainscore
    "yuanli2333": yuanli2333
    "rsatoolbox": rsatoolbox
    "scipy": scipy
    "sim_metric": sim_metric
    "svcca": svcca
}

// default backend choice for each metric
// id instead of name? e.g. _default_backend: [#MetricId]: #BackendId  // TODO?
#default_backend: [#MetricName]: #BackendName  // schema
#default_backend: {
    procrustes: "netrep"
    cca: "netrep"
    svcca: "netrep"
    pwcca: "sim_metric"
    cca_mean_sq_corr: "sim_metric"
    cka: "yuanli2333"
    rsa: "rsatoolbox"
    linear_regression: "brainscore"
    pls: "svcca"
}
// TODO: extract names from type #MetricName?
// used in metric.cue to create the metric fields


#metric_names: backend.#metric_names
#MetricName: backend.#MetricName
// #MetricName: similarity.#MetricName
// #BackendName: similarity.#BackendName
#BackendName: or([for id, _ in #backends { id }])

cards: {
    for id, backend in #backends {
        (id): {
            backend.card
            metrics: [for k, v in backend.metric { k }]
        }
    }
}
metric_names: #metric_names

metric_by_backend: #BackendName: [...#MetricName]
metric_by_backend: {
    for id, backend in #backends {
        (id): [for k, v in backend.metric { k }]
    }
}

backend_by_metric: #MetricName: [...#BackendName]
backend_by_metric: {
    for metric_name in #metric_names {
        (metric_name): [
            for backend_name, metrics in metric_by_backend if list.Contains(metrics, metric_name) {
                backend_name
            }
        ]
    }
}

backends: #backends
default_backend: #default_backend

