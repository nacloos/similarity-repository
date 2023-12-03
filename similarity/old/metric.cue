// // measure is the main interface to use a measure's default implementation
// package measure
// import(
//     "github.com/similarity/backend:backends"
// )

// _backends: backends.#backends
// _default_backend: backends.#default_backend
// _measure_names: backends.#measure_names

// // TODO: do that in api.cue?
// // create fields for measures that have a default implementation
// for name in _measure_names if _default_backend[name] != _|_ {
//     // TODO: "let" statement seems to terribly slow down the compilation
//     // let backend = _backends[_default_backend[name]]
//     // (name): backend.measure[name]
//     // much faster than the two lines above!
//     (name): _backends[_default_backend[name]].measure[name]
// }
