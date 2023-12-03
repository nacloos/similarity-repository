package backend
import(
    // TODO: might be confusing to use that now
    // "github.com/netrep/metrics"
    "math"
)


// TODO: separate distance and score
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
            #path: "netrep.metrics.Linearmeasure"
            #preprocessing: [#reshape2d]
            "alpha": alpha
            score_method: "angular"
        }
        "shape_metric-euclidean-alpha\(math.Round(alpha*10))e-1": {
            #path: "netrep.metrics.Linearmeasure"
            #preprocessing: [#reshape2d]
            "alpha": alpha
            score_method: "euclidean"
        }
    }

    // procrustes: measures.#Linearmeasure & {alpha: 1}
    procrustes: {
        #path: "netrep.metrics.Linearmeasure"
        alpha: 1
    }
    // "procrustes-sq": procrustes & {
    //     #postprocessing: [
    //         // square the distance
    //         #target & {#path: "similarity.processing.square_score", #partial: true}
    //     ]
    // }
    // cca: measures.#Linearmeasure & {alpha: 0}
    "cca-angular": {
        #path: "netrep.metrics.Linearmeasure"
        #preprocessing: [#reshape2d]
        alpha: 0
        score_method: "angular"
    }
    "cca-euclidean": {
        #path: "netrep.metrics.Linearmeasure"
        #preprocessing: [#reshape2d]
        alpha: 0
        score_method: "euclidean"
    }
    // svcca or svcca angular?
    // svcca: measures.#Linearmeasure & {
    svcca: {
        #path: "netrep.metrics.Linearmeasure"
        alpha: 0
        #preprocessing: [
            #reshape2d, 
            #pca & {n_components: 0.95}
        ]
    }
    // permutation: measures.#Permutationmeasure
    permutation: {
        #path: "netrep.metrics.Permutationmeasure"
    }



    "svcca-var95": svcca
    // "svcca-var99": measures.#Linearmeasure & {
    "svcca-var99": {
        #path: "netrep.metrics.Linearmeasure"
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


card: {
    name: "Generalized Shape measures on Neural Representations"
    github: "https://github.com/ahwillia/netrep"
    citation: [
"""
@inproceedings{neural_shape_measures,
    author = {Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman},
    title = {Generalized Shape measures on Neural Representations},
    year = {2021},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {34},
}
""",
"""
@inproceedings{stochastic_neural_shape_measures,
    author = {Lyndon R. Duong and Jingyang Zhou and Josue Nassar and Jules Berman and Jeroen Olieslagers and Alex H. Williams},
    title = {Representational dissimilarity measure spaces for stochastic neural networks},
    year = {2023},
    booktitle = {International Conference on Learning Representations},
}
"""
    ]
}