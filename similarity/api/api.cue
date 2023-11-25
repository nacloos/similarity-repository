package api
import(
    "github.com/similarity/metric"
    "github.com/similarity/backend:backends"
)

// TODO: openapi schema?
"metric": {
    // TODO: temp
    for k, v in metric if (k != "papers" && k != "cards") {
        (k): v
    }
}
"backend": backends.#backends
"paper": metric.papers
