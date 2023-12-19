package backends
import(
    "list"
    "github.com/similarity/measure"

    netrep              "github.com/similarity/backend/netrep:backend"
    repsim              "github.com/similarity/backend/repsim:backend"
    brainscore          "github.com/similarity/backend/brainscore:backend"
    yuanli2333          "github.com/similarity/backend/yuanli2333:backend"
    rsatoolbox          "github.com/similarity/backend/rsatoolbox:backend"
    mklabunde           "github.com/similarity/backend/mklabunde:backend"
    sim_metric          "github.com/similarity/backend/sim_metric:backend"
    svcca               "github.com/similarity/backend/svcca:backend"
    imd                 "github.com/similarity/backend/imd:backend"
    subspacematch       "github.com/similarity/backend/subspacematch:backend"
    nn_similarity_index "github.com/similarity/backend/nn_similarity_index:backend"
)
#measure_ids: measure.#measure_ids
#MeasureId: measure.#MeasureId

_backends: [string]: _
_backends: {
    // will validate the backends
    "netrep":               netrep
    "repsim":               repsim
    "brainscore":           brainscore
    "yuanli2333":           yuanli2333
    "rsatoolbox":           rsatoolbox
    "mklabunde":            mklabunde
    "sim_metric":           sim_metric
    "svcca":                svcca
    "imd":                  imd
    "subspacematch":        subspacematch
    "nn_similarity_index": nn_similarity_index
}

// define the backend id type based on the given backends
#BackendId: or([for id, _ in _backends { id }])


// default backend choice for each measure
#default_backend: [#MeasureId]: #BackendId  // schema
// TODO: if a backend is the only one that supports a measure, then it should be the default backend for that measure
#default_backend: {
    procrustes:         "netrep"
    cca:                "netrep"
    svcca:              "netrep"
    "svcca-var95":      "netrep"
    "svcca-var99":      "netrep"
    permutation:        "netrep"

    // TODO: temp
    for id, _ in netrep.measure if id != "cka" {
        (id): "netrep"
    }

    pwcca:              "sim_metric"
    cca_mean_sq_corr:   "sim_metric"
    "procrustes-sq-euclidean": "sim_metric"

    cka:                "yuanli2333"
    nbs:                "nn_similarity_index"

    "riemannian_metric": "repsim"
    
    // TODO
    // rsa:                "rsatoolbox"
    // [string & =~ "^rsa.*"]: "rsatoolbox"
    
    // all the measures in rsatoolbox that starts with rsa
    for id, _ in rsatoolbox.measure if id =~ "^rsa.*"{
        (id): "rsatoolbox"
    }

    linear_regression:  "brainscore"
    correlation:        "brainscore"

    pls:                "svcca"
    imd:                "imd"
    max_match:          "subspacematch"
}


// automaticallly add derived measures to each backend if it is not already defined
// e.g. if a backend implements cka it also automatically implements cka-angular (just take the cos of cka)
#derived_measures: {
    backend_name: #BackendId
    backend: _
    transforms: [...]

    out: {
        for k, v in backend.measure
        for T in transforms 
        if T.inp == k && backend.measure[T.out] == _|_ {
        // && out[T.out] == _|_ {
            (T.out): {
                v
                #_postprocessing: T.function
            }
            // TODO: conflicting list lengths errror
            // (T.out): "_out_": {
            //     for kk, vv in v["_out_"] if kk != "postprocessing" {
            //         (kk): vv
            //     }
            //     // append tsf to postprocessing
            //     "postprocessing": v["_out_"]["postprocessing"] + T.function
            // }
        }
    }
}

// #derive_measures: test_transforms.#derive_measures
// TODO: structural cycle if use #backends in for loop
#backends: {
    for backend_name, backend in _backends {
        (backend_name): "measure": {
            (#derived_measures & {
                "backend_name": backend_name
                "backend": backend
                "transforms": measure.transforms
            }).out
        }
        (backend_name): backend
    }
}
// #backends: _backends

measure_ids: #measure_ids

measure_by_backend: #BackendId: [...#MeasureId]
measure_by_backend: {
    for id, backend in #backends {
        (id): [for k, v in backend.measure { k }]
    }
}

backend_by_measure: #MeasureId: [...#BackendId]
backend_by_measure: {
    for measure_name in #measure_ids {
        (measure_name): [
            for backend_name, measures in measure_by_backend if list.Contains(measures, measure_name) {
                backend_name
            }
        ]
    }
}

backends: #backends
default_backend: #default_backend



measures: {
    // create fields for measures that have a default implementation
    for name in measure_ids if default_backend[name] != _|_ {
        // TODO: "let" statement seems to terribly slow down the compilation
        // let backend = _backends[_default_backend[name]]
        // (name): backend.measure[name]
        // (name): backends[default_backend[name]].measure[name]

        if backends[default_backend[name]].measure[name] != _|_ {
            (name): backends[default_backend[name]].measure[name]
        } 
    }
}
