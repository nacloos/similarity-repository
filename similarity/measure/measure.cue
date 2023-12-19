package measure
import(
    "math"
    "strings"
    "github.com/similarity/utils"
    "github.com/similarity/papers"
)
#Paper: papers.#Paper

// #MeasureId type is defined here
#measure_ids: [for k, _ in cards { k }]
#MeasureId: or(#measure_ids)

// a card describe a measure independently of any implementation
#MeasureCard: {
    name?: string
    paper?: #Paper | [...#Paper]

    // only string because used to generate metric id (TODO: allow number and convert to string?)
    parameters?: [string]: [...string]
    defaults?: _
    naming?: string

    properties: [...#PropertyId] | *[]
    backends: _ | *[]
    default_backend: _ | *null

    // python implementation using default backend
    "_out_"?: _
}

#PropertyId: or([for k, v in property { k }])

property: [string]: {
    name?: string
    description?: string
}
property: {
    "permutation_invariant": {
        name: "Permutation Invariant"
    },
    "scale_invariant": {
        name: "Scale Invariant"
    }
    "rotation_invariant": {
        name: "Rotation Invariant"
    }
    "invertible_linear_invariant": {
        name: "Invertible Linear Invariant"
    }
    "translation_invariant": {
        name: "Translation Invariant"
    }
    "affine_invariant": {
        name: "Affine Invariant"
    }
    score: {
        name: "Score"
        description: "Measure of similarity. A score of 1 indicates that the representations are identical."
    }
    metric: {
        name: "Metric"
        description: "Measure that satisfies the axioms for a metric: non-negativity, identity, symmetry, and triangle inequality. A metric of 0 indicates that the representations are identical."
    }
}

_cards: [string]: #MeasureCard
_cards: {
    ...
    euclidean: {}  // TODO: temp
    permutation: {
        name: "Permutation"
        parameters: {
            score_method: ["euclidean", "angular"]
        }
        properties: [
            "permutation_invariant",
            "metric"
        ]
    }
    correlation: {
        name: "Correlation"
    } 
    // canonical correlation analysis
    cca: {  // TODO: call it cca or mean_cca?
        // TODO: klabunde23 survey has two rows for mean cc, ok to merge them here?
        name: "Mean Canonical Correlation"
        paper: [papers.yanai1974, papers.raghu2017, papers.kornblith2019]
        parameters: {
            scoring_method: ["euclidean", "angular"]
        }
        properties: [
            "permutation_invariant",
            "scale_invariant",
            "rotation_invariant",
            "invertible_linear_invariant",
            "translation_invariant",
            "affine_invariant",
            "score",
        ]
    }
    cca_mean_sq_corr: {
        name: "Mean Squared Canonical Correlation"
        // properties: [
        //     "score"
        // ]
    }

    svcca: {
        name: "Singular Vector Canonical Correlation Analysis"
        paper: papers.raghu2017
        parameters: {
            variance_fraction: ["var95", "var99"]
        }
        properties: [
            "permutation_invariant",
            "scale_invariant",
            "rotation_invariant",
            "translation_invariant",
            "score",
        ]
    }
    pwcca: {
        name: "Projection-Weighted Canonical Correlation Analysis"
        paper: papers.morcos2018
        properties: [
            "scale_invariant",
            "translation_invariant",
            "score",
        ]
    }

    "riemannian_metric": {
        name: "Riemannian Metric"
        paper: papers.shahbazi2021
        properties: [
            "metric"
        ]
    }

    procrustes: {
        name: "Orthogonal Procrustes"
        paper: [papers.ding2021, papers.williams2021]
        parameters: {
            scoring_method: ["euclidean", "angular"]
        }
        // properties: [
        //     "metric",
        //     "permutation_invariant",
        //     "scale_invariant",
        //     "rotation_invariant"
        // ]
    }
    "procrustes-euclidean": properties: [
        "permutation_invariant",
        "rotation_invariant",
        "metric",
    ]
    "procrustes-angular": properties: [
        "permutation_invariant",
        "rotation_invariant",
        "scale_invariant",
        "metric",
    ]
    // TODO: use argument? e.g. squared_or_not: ["sq", null]
    "procrustes-sq": procrustes

    "procrustes-score": {
        name: "Procrustes Score"
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "score"
        ]
    }

    for alpha in [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0] {
        "shape_metric-angular-alpha\(math.Round(alpha*10))e-1": {
            name: "Angular Shape Metric"
            paper: papers.williams2021
            // TODO: alpha=0 equivalant to cca, alpha=1 to procrustes
            // parameters: {
            //     alpha: ["alpha0", "alpha0.5", "alpha1"]
            // }
            properties: ["metric"]
        }
        "shape_metric-euclidean-alpha\(math.Round(alpha*10))e-1": {
            name: "Euclidean Shape Metric"
            paper: papers.williams2021
            properties: ["metric"]
        }
    }
    // partial_whitening_shape_metric: {
    //     name: "Partial-Whitening Shape Metric"
    //     paper: papers.williams2021
    // }
    // TODO
    pls: {
        name: "Partial Least Squares"
        properties: [
            "score"
        ]
        // paper: TODO
    }
    linear_regression: {
        name: "Linear Regression"
        paper: [papers.li2016, papers.kornblith2019]
        properties: [
            "score"
        ]
    }
    aligned_cosine: {
        name: "Aligned Cosine Similarity"
        paper: papers.hamilton2016a
    }
    corr_match: {
        name: "Correlation Match"
        paper: papers.li2016
    }
    max_match: {
        name: "Maximum Match"
        paper: papers.wang2018
    }

    // representational similarity matrix
    rsm_norm: {
        name: "Representational Similarity Matrix Norms"
        paper: [papers.shahbazi2021, papers.yin2018]
    }

    rsa: {
        name: "Representational Similarity Analysis"
        paper: papers.kriegeskorte2008
        properties: [
            "permutation_invariant",
            "scale_invariant",
            "translation_invariant",
            "score"
        ]
        parameters: {
            rdm_method: [
                "euclidean",
                "correlation",
                "mahalanobis",
                "crossnobis",
                "poisson",
                "poisson_cv"
            ]
            compare_method: [
                "cosine",
                "spearman",
                "corr",
                "kendall",
                "tau_b",
                "tau_a",
                "rho_a",
                "corr_cov",
                "cosine_cov",
                "neg_riem_dist"
            ]
        }
        // TODO: provide defaults for "rsa"
        defaults: {
            rdm_method: "euclidean"
            compare_method: "cosine"
        }
        // naming: "rsa-{rdm_method}-{compare_method}"
    }    

    cka: {
        name: "Centered Kernel Alignment"
        paper: papers.kornblith2019
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "translation_invariant",
            "score"
        ]
    }
    "cka-angular": {
        name: "Angular CKA"
        paper: [papers.williams2021, papers.lange2022]
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "translation_invariant",
            "metric"
        ]
    }
    dcor: {
        name: "Distance Correlation"
        paper: papers.szekely2007
    }
    nbs: {
        name: "Normalized Bures Similarity"
        paper: papers.tang2020
    }
    eos: {
        name: "Eigenspace Overlap Score"
        paper: papers.may2019
    }
    gulp: {
        name: "Unified Linear Probing"
        paper: papers.boixadsera2022
    }
    "riemmanian_metric": {
        name: "Riemmanian Distance"
        paper: papers.shahbazi2021
        properties: [
            "metric"
        ]
    }

    // neighbors
    jaccard: {
        name: "Jaccard"
        paper: [papers.schumacher2021, papers.wang2020, papers.hryniowski2020, papers.gwilliam2022]
    }
    second_order_cosine: {
        name: "Second-Prder Cosine Similarity"
        paper: papers.hamilton2016b
    }
    rank_similarity: {
        name: "Rank Similarity"
        paper: papers.wang2020
    }
    joint_rank_jaccard: {
        name: "Joint Rang and Jaccard Similarity"
        paper: papers.wang2020
    }

    // topology
    gs: {
        name: "Geometry Score"
        paper: papers.khrulkov2018
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "translation_invariant",
        ]
    }
    imd: {
        name: "Multi-scale Intrinsic Distance"
        paper: papers.tsitsulin2020
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "translation_invariant",
        ]
    }
    rtd: {
        name: "Representation Topology Divergence"
        paper: papers.barannikov2022
        properties: [
            "permutation_invariant",
            "rotation_invariant",
            "scale_invariant",
            "translation_invariant",
        ]
    }

    // statistic
    intrinsic_dimension: {
        name: "Intrinsic Dimension"
        paper: papers.camastra2016
    }
    magnitude: {
        name: "Magnitude"
        paper: papers.wang2020
    }
    concentricity: {
        name: "Concentricity"
        paper: papers.wang2020
    }
    uniformity: {
        name: "Uniformity"
        paper: papers.wang2022
    }
    tolerance: {
        name: "Tolerance"
        paper: papers.wang2021
    }
    knn_graph_modularity: {
        name: "kNN-Graph Modularity"
        paper: papers.lu2022
    }
    neuron_graph_modularity: {
        name: "Neuron-Graph Modularity"
        paper: papers.lange2022
    }
}

// derive cards by taking the cartesian product of the parameters
cards: {
    for key, card in _cards {
        if card.parameters == _|_ { (key): card }
        if card.parameters != _|_ {
            for p in (utils.#Cartesian & {inp: card.parameters}).out {
                let metric_name = key + "-" + strings.Join(p, "-")
                (metric_name): card
            }
            // TODO: default params
            (key): card
        }
    }
}

