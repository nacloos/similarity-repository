// package card
import(
    "strings"
    "github.com/similarity/utils"
)

_rdm_methods: [
    "euclidean",
    "correlation",
    "mahalanobis",
    "crossnobis",
    "poisson",
    "poisson_cv"
]
// TODO: no '-' in names, only '_' (but then need to convert name when calling rsatoolbox)    #compare_methods: [
_compare_methods: [
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

cards: {
    // for rdm_method in _rdm_methods for compare_method in _compare_methods {
    //     ("rsa-" + rdm_method + "-" + compare_method): {
    //         name: "RSA RDM-\(rdm_method) Compare-\(compare_method)"
    //         paper: papers.kriegeskorte2008
    //     }
    // }   
    // default
    rsa: cards["rsa-euclidean-cosine"]
    

    _params: {
        rdm_method: _rdm_methods
        compare_method: _compare_methods
    }
    for p in (utils.#Cartesian & {inp: _params}).out {
        let name = "rsa-" + strings.Join(p, "-")
        (name): {
            "name": name
            paper: papers.kriegeskorte2008
        }
    }

    // TODO: get either all variations of just the default params
    // rsa: {
    //     name: "Representational Similarity Analysis"
    //     paper: papers.kriegeskorte2008
    //     parameters: {
    //         // order?
    //         rdm_method: _rdm_methods
    //         compare_method: _compare_methods
    //     } // automatically create variations?
    //     // TODO: provide defaults for "rsa"
    //     defaults: {
    //         rdm_method: "euclidean"
    //         compare_method: "cosine"
    //     }
    //     naming: "rsa-{rdm_method}-{compare_method}"
    // }
}
