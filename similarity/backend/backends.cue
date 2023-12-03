// can't use the backend package because would cause circular import
package backends
import(
    "list"
    // "github.com/similarity"
    "github.com/similarity/backend"
    // "github.com/similarity/measure:test_transforms"
    measure_cards "github.com/similarity/measure:card"

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
// automaticallly add derived measures to each backend if it is not already defined
// need to write somewhere the transformation pipeline from "cka" to "cka-angular"

#derived_measures: {
    backend_name: #BackendName
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

// #derive_measures: test_transforms.#derive_measures
// TODO: structural cycle if use #backends in for loop
#backends: {
    for backend_name, backend in _backends {
    // for backend_name, backend in {"sim_metric": sim_metric} {
        (backend_name): measure: {
            // slower
            // (#derive_measures & {
            //     measures: backend.measure
            //     transforms: measure_cards.transforms
            //     max_depht: 0
            // }).out
            (#derived_measures & {
                "backend_name": backend_name
                "backend": backend
                transforms: measure_cards.transforms
            }).out
        }
        (backend_name): backend
    }
}
// #backends: _backends
// _a: measure_cards

// _backend: sim_metric
// for k, v in _backend.measure {
//     for T in measure_cards.transforms 
//     if T.inp == k && _backend.measure[T.out] == _|_ {
//         #backends: sim_metric: measure: (T.out): {
//             _backend.measure[k]
//             // TODO: add T.function to postprocessing
//         }
//     }
// }

// default backend choice for each measure
// id instead of name? e.g. _default_backend: [#measureId]: #BackendId  // TODO?
#default_backend: [#MeasureName]: #BackendName  // schema
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

// all the measures that have a card
#measure_names: backend.#measure_names

#MeasureName: backend.#MeasureName
#BackendName: or([for id, _ in #backends { id }])

cards: {
    // backend cards
    for id, backend in #backends {
        (id): {
            backend.card
            measures: [for k, v in backend.measure { k }]
        }
    }
}
measure_names: #measure_names

measure_by_backend: #BackendName: [...#MeasureName]
measure_by_backend: {
    for id, backend in #backends {
        (id): [for k, v in backend.measure { k }]
    }
}

backend_by_measure: #MeasureName: [...#BackendName]
backend_by_measure: {
    for measure_name in #measure_names {
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
    for name in measure_names if default_backend[name] != _|_ {
        // TODO: "let" statement seems to terribly slow down the compilation
        // let backend = _backends[_default_backend[name]]
        // (name): backend.measure[name]
        // much faster than the two lines above!
        (name): backends[default_backend[name]].measure[name]
    }
}
