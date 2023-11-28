// metric is the main interface to use a metric's default implementation
package metric
import(
    "github.com/similarity/backend:backends"
)

_backends: backends.#backends
_default_backend: backends.#default_backend
_metric_names: backends.#metric_names

// TODO: do that in api.cue?
// create fields for metrics that have a default implementation
for name in _metric_names if _default_backend[name] != _|_ {
    // TODO: "let" statement seems to terribly slow down the compilation
    // let backend = _backends[_default_backend[name]]
    // (name): backend.metric[name]
    // much faster than the two lines above!
    (name): _backends[_default_backend[name]].metric[name]
}
