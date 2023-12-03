package backend
import(
    "math"
    "github.com/similarity/papers"
)

github: "https://github.com/ahwillia/netrep"
paper: [papers.williams2021]


measure: {
    [string]: #Measure & {
        // #postprocessing: [
        //     #angular_dist_to_score
        // ]
    }

    [("procrustes" | "cca" | "cka" | "cka-angular" | "permutation")]: {
        #preprocessing: [
            #reshape2d
        ]
    }

    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        // TODO: scientific notation (dot used to separate hierarchical levels)
        "shape_metric-angular-alpha\(math.Round(alpha*10))e-1": {
            #path: "netrep.metrics.LinearMetric"
            #preprocessing: [#reshape2d]
            "alpha": alpha
            score_method: "angular"
        }
        "shape_metric-euclidean-alpha\(math.Round(alpha*10))e-1": {
            #path: "netrep.metrics.LinearMetric"
            #preprocessing: [#reshape2d]
            "alpha": alpha
            score_method: "euclidean"
        }
    }

    procrustes: {
        #path: "netrep.metrics.LinearMetric"
        alpha: 1
    }
    // "procrustes-sq": procrustes & {
    //     #postprocessing: [
    //         // square the distance
    //         #target & {#path: "similarity.processing.square_score", #partial: true}
    //     ]
    // }
    "cca-angular": {
        #path: "netrep.metrics.LinearMetric"
        #preprocessing: [#reshape2d]
        alpha: 0
        score_method: "angular"
    }
    "cca-euclidean": {
        #path: "netrep.metrics.LinearMetric"
        #preprocessing: [#reshape2d]
        alpha: 0
        score_method: "euclidean"
    }
    // svcca or svcca angular?
    svcca: {
        #path: "netrep.metrics.LinearMetric"
        alpha: 0
        #preprocessing: [
            #reshape2d, 
            #pca & {n_components: 0.95}
        ]
    }
    permutation: {
        #path: "netrep.metrics.PermutationMetric"
    }



    "svcca-var95": svcca
    "svcca-var99": {
        #path: "netrep.metrics.LinearMetric"
        alpha: 0
        #preprocessing: [
            #reshape2d, 
            #pca & {n_components: 0.99}
        ]
    }

    // for key in ["procrustes", "procrustes-sq", "cca", "permutation"] {
    for key in ["procrustes", "permutation"] {
        for score_method in ["euclidean", "angular"] {
            (key +"-" + score_method): {
                measure[key]
                "score_method": score_method
            }
        }
    }

    "cka-angular": {
        #path: "similarity.backend.netrep.cka.LinearCKA"
        // netrep LinearCKA doesn't have a fit_score method
        #call_key: "score"
    }
    cka: measure["cka-angular"] & {
        #postprocessing: [
            #target & {#path: "similarity.processing.cosine_score", #partial: true}
        ]
    }
}

