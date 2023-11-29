package backend
import(
    // TODO: might be confusing to use that now
    "github.com/netrep/metrics"
)


// TODO: separate distance and score
metric: {
    [string]: #Metric & {
        // #postprocessing: [
        //     #angular_dist_to_score
        // ]
    }

    [("procrustes" | "cca" | "cka" | "permutation")]: {
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
    "svcca-var95": svcca
    "svcca-var99": metrics.#LinearMetric & {
        alpha: 0
        #preprocessing: [
            #reshape2d, 
            #pca & {n_components: 0.99}
        ]
    }
    
    permutation: metrics.#PermutationMetric
    for score_method in ["euclidean", "angular"] {
        ("permutation-" + score_method): metrics.#PermutationMetric & {
            "score_method": score_method
        }
    }
    // bug in netrep LinearCKA: "NameError: name 'angular_distance' is not defined"
    // cka: metrics.#LinearCKA & {
    //     // netrep LinearCKA doesn't have a fit_score method
    //     #call_key: "score"
    // }
}


card: {
    name: "Generalized Shape Metrics on Neural Representations"
    github: "https://github.com/ahwillia/netrep"
    citation: [
"""
@inproceedings{neural_shape_metrics,
    author = {Alex H. Williams and Erin Kunz and Simon Kornblith and Scott W. Linderman},
    title = {Generalized Shape Metrics on Neural Representations},
    year = {2021},
    booktitle = {Advances in Neural Information Processing Systems},
    volume = {34},
}
""",
"""
@inproceedings{stochastic_neural_shape_metrics,
    author = {Lyndon R. Duong and Jingyang Zhou and Josue Nassar and Jules Berman and Jeroen Olieslagers and Alex H. Williams},
    title = {Representational dissimilarity metric spaces for stochastic neural networks},
    year = {2023},
    booktitle = {International Conference on Learning Representations},
}
"""
    ]
}