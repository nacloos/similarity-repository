package api
import(
    "github.com/similarity/metric"
    "github.com/similarity/backend:backends"
)

// TODO: openapi schema?
"metric": metric
"backend": backends.#backends
"paper": metric.papers
