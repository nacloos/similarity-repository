package backends
import(
    "github.com/similarity/measure"
)

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
#apply_transforms_once: self={
    backends: _
    out: {
        for backend_name, backend in self.backends {
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
}

#derive_measures_in_backends: self={
    backends: _
    // TODO: make it more general
    // apply transforms a fixed number of times
    out1: (#apply_transforms_once & {backends: self.backends}).out
    // out2: (#apply_transforms_once & {backends: self.out1}).out
    // out3: (#apply_transforms_once & {backends: self.out2}).out
    out: out1
}
