// can't use the backend package because would cause circular import
package backends
import(
    "github.com/similarity"
    netrep "github.com/similarity/backend/netrep:backend"
    yuanli2333 "github.com/similarity/backend/yuanli2333:backend"
    rsatoolbox "github.com/similarity/backend/rsatoolbox:backend"
)

#MetricName: similarity.#MetricName
#BackendName: similarity.#BackendName

// TODO: extract names from type #MetricName?
#metric_names: ["procrustes", "cca", "svcca", "cka", "rsa"]


#backends: {
    // will validate the backends
    "netrep": netrep
    "yuanli2333": yuanli2333
    "rsatoolbox": rsatoolbox
}

// default backend choice for each metric
#default_backend: [#MetricName]: #BackendName
// _default_backend: [#MetricId]: #BackendId  // TODO?
#default_backend: {
    procrustes: "netrep"
    cca: "netrep"
    svcca: "netrep"
    cka: "yuanli2333"
    rsa: "rsatoolbox"
}
