package backend
import(
    "github.com/netrep/metrics"
)


// TODO: separate distance and score
metric: {
    [string]: #Metric & {
        #postprocessing: [
            #angular_dist_to_score
        ]
    }

    [("procrustes" | "cca" | "cka")]: {
        #preprocessing: [
            #reshape2d
        ]
    }

    procrustes: metrics.#LinearMetric & {alpha: 1}
    cca: metrics.#LinearMetric & {alpha: 0}
    svcca: metrics.#LinearMetric & {
        alpha: 0
        #preprocessing: [
            #reshape2d, 
            #pca & {n_components: 0.95}
        ]
    }
    cka: metrics.#LinearCKA
}



