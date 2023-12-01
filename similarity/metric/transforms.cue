package card
import(
    "github.com/similarity/utils"
)
#target: utils.#target


#score_tsf: #target & {
    #in_keys: ["score"]
    #out_keys: ["score"]
}
#cos: #score_tsf & {
    #path: "similarity.processing.cosine_score"
    #partial: true
}
#arccos: #score_tsf & {
    #path: "similarity.processing.arccos_score"
    #partial: true
}
#square: #score_tsf & {
    #path: "similarity.processing.square_score"
    #partial: true
}
#sqrt: #score_tsf & {
    #path: "similarity.processing.sqrt_score"
    #partial: true
}
#angular_to_euclidean_shape_metric: #target & {
    #path: "similarity.processing.angular_to_euclidean_shape_metric"
    #partial: true
    #in_keys: ["X", "Y", "score"]
    #out_keys: ["score"]
}
#euclidean_to_angular_shape_metric: #target & {
    #path: "similarity.processing.euclidean_to_angular_shape_metric"
    #partial: true
    #in_keys: ["X", "Y", "score"]
    #out_keys: ["score"]
}


// TODO: compose tsfs?
transforms: [
    {inp: "cka", out: "cka-angular", function: [#arccos]},
    {inp: "cka-angular", out: "cka", function: [#cos]},

    {inp: "cca", out: "cca-angular", function: [#arccos]},
    {inp: "cca-angular", out: "cca", function: [#cos]},

    // {inp: "procrustes", out: "procrustes-sq", function: [#square]},
    // {inp: "procrustes-sq", out: "procrustes", function: [#sqrt]},
    // TODO: structural cycle
    // Squared procrustes only for euclidean because squared dist has been used only in the euclidean case
    {inp: "procrustes-euclidean", out: "procrustes-sq-euclidean", function: [#square]},
    {inp: "procrustes-sq-euclidean", out: "procrustes-euclidean", function: [#sqrt]},

    {inp: "procrustes-euclidean", out: "procrustes-angular", function: [#euclidean_to_angular_shape_metric]},
    {inp: "procrustes-angular", out: "procrustes-euclidean", function: [#angular_to_euclidean_shape_metric]},

    // {inp: "procrustes-angular", out: "procrustes-score", function: []}

    // TODO: use recursion?
    {inp: "procrustes-sq-euclidean", out: "procrustes-angular", function: [#sqrt, #euclidean_to_angular_shape_metric]},

    // Identity tsf for same metrics that have different name
    // TODO: two-way transforms
    // {inp: "shape_metric-angular-alpha1", out: "procrustes-angular"},
    // {out: "shape_metric-angular-alpha1", inp: "procrustes-angular"},
    // {inp: "shape_metric-angular-alpha0", out: "cca-angular"},
    // {out: "shape_metric-angular-alpha0", inp: "cca-angular"},
]
// transforms: [...{inp: #MetricName, out: #MetricName, function?: [...]}]
