// metric is the main interface to use a metric's default implementation
package metric
import(
    "github.com/similarity/backend:backends"
)

_backends: backends.#backends
_default_backend: backends.#default_backend
_metric_names: backends.#metric_names

// create the metric fields
for name in _metric_names {
    let backend = _backends[_default_backend[name]]
    (name): backend.metric[name]
}
