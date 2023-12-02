// can't use the backend package because would cause circular import
package backends
import(
    "list"
    // "github.com/similarity"
    "github.com/similarity/backend"
    // "github.com/similarity/metric:test_transforms"
    metric_cards "github.com/similarity/metric:card"

    netrep          "github.com/similarity/backend/netrep:backend"
    repsim          "github.com/similarity/backend/repsim:backend"
    brainscore      "github.com/similarity/backend/brainscore:backend"
    yuanli2333      "github.com/similarity/backend/yuanli2333:backend"
    rsatoolbox      "github.com/similarity/backend/rsatoolbox:backend"
    mklabunde       "github.com/similarity/backend/mklabunde:backend"
    sim_metric      "github.com/similarity/backend/sim_metric:backend"
    svcca           "github.com/similarity/backend/svcca:backend"
    imd             "github.com/similarity/backend/imd:backend"
    subspacematch   "github.com/similarity/backend/subspacematch:backend"
)

_backends: [string]: _  // schema
_backends: {
    // will validate the backends
    "netrep":           netrep
    "repsim":           repsim
    "brainscore":       brainscore
    "yuanli2333":       yuanli2333
    "rsatoolbox":       rsatoolbox
    "mklabunde":        mklabunde
    "sim_metric":       sim_metric
    "svcca":            svcca
    "imd":              imd
    "subspacematch":    subspacematch
}
// TODO: if a backend implements cka it also automatically implements cka-angular (just take the cos of cka)
// automaticallly add derived metrics to each backend if it is not already defined
// need to write somewhere the transformation pipeline from "cka" to "cka-angular"

#derived_metrics: {
    backend_name: #BackendName
    backend: _
    transforms: [...]

    out: {
        for k, v in backend.metric
        for T in transforms 
        if T.inp == k && backend.metric[T.out] == _|_ {
        // && out[T.out] == _|_ {
            (T.out): {
                v
                #_postprocessing: T.function
                // TODO: not working
                // for kk, vv in v if kk != "_postprocessing_" {
                //     (kk): vv
                // }
                // // append tsf to postprocessing
                // "_postprocessing_": v["_postprocessing_"] + T.function
            }
        }
    }
}

// #derive_metrics: test_transforms.#derive_metrics
// TODO: structural cycle if use #backends in for loop
#backends: {
    for backend_name, backend in _backends {
    // for backend_name, backend in {"sim_metric": sim_metric} {
        (backend_name): metric: {
            // slower
            // (#derive_metrics & {
            //     metrics: backend.metric
            //     transforms: metric_cards.transforms
            //     max_depht: 0
            // }).out
            (#derived_metrics & {
                "backend_name": backend_name
                "backend": backend
                transforms: metric_cards.transforms
            }).out
        }
        (backend_name): backend
    }
}
// #backends: _backends
// _a: metric_cards

// _backend: sim_metric
// for k, v in _backend.metric {
//     for T in metric_cards.transforms 
//     if T.inp == k && _backend.metric[T.out] == _|_ {
//         #backends: sim_metric: metric: (T.out): {
//             _backend.metric[k]
//             // TODO: add T.function to postprocessing
//         }
//     }
// }

// default backend choice for each metric
// id instead of name? e.g. _default_backend: [#MetricId]: #BackendId  // TODO?
#default_backend: [#MetricName]: #BackendName  // schema
// TODO: if a backend is the only one that supports a metric, then it should be the default backend for that metric
#default_backend: {
    procrustes:         "netrep"
    cca:                "netrep"
    svcca:              "netrep"
    "svcca-var95":      "netrep"
    "svcca-var99":      "netrep"
    permutation:        "netrep"

    // TODO: temp
    for id, _ in netrep.metric if id != "cka" {
        (id): "netrep"
    }

    pwcca:              "sim_metric"
    cca_mean_sq_corr:   "sim_metric"
    "procrustes-sq-euclidean": "sim_metric"

    cka:                "yuanli2333"

    // TODO
    // rsa:                "rsatoolbox"
    // [string & =~ "^rsa.*"]: "rsatoolbox"
    
    // all the metrics in rsatoolbox that starts with rsa
    for id, _ in rsatoolbox.metric if id =~ "^rsa.*"{
        (id): "rsatoolbox"
    }

    linear_regression:  "brainscore"
    correlation:        "brainscore"

    pls:                "svcca"
    imd:                "imd"
    max_match:          "subspacematch"
}

// all the metrics that have a card
#metric_names: backend.#metric_names

#MetricName: backend.#MetricName
#BackendName: or([for id, _ in #backends { id }])

cards: {
    // backend cards
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

