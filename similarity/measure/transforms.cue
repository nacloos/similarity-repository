package measure
import(
    "github.com/similarity/utils"
)
#target: utils.#target


#score_tsf: #target & {
    #in_keys: ["score"]
    #out_keys: ["score"]
    #partial: true
}
#cos: #score_tsf & {
    #path: "similarity.processing.cosine_score"
}
#arccos: #score_tsf & {
    #path: "similarity.processing.arccos_score"
}
#square: #score_tsf & {
    #path: "similarity.processing.square_score"
}
#sqrt: #score_tsf & {
    #path: "similarity.processing.sqrt_score"
}
#angular_dist_to_score: #score_tsf & {
    #path: "similarity.processing.angular_dist_to_score"
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
    // composition: cca => cca-angular => cca-euclidean
    {inp: "cca", out: "cca-euclidean", function: [#cos, #angular_to_euclidean_shape_metric]},

    // {inp: "procrustes", out: "procrustes-sq", function: [#square]},
    // {inp: "procrustes-sq", out: "procrustes", function: [#sqrt]},
    // TODO: structural cycle
    // Squared procrustes only for euclidean because squared dist has been used only in the euclidean case
    {inp: "procrustes-euclidean", out: "procrustes-sq-euclidean", function: [#square]},
    {inp: "procrustes-sq-euclidean", out: "procrustes-euclidean", function: [#sqrt]},

    {inp: "procrustes-euclidean", out: "procrustes-angular", function: [#euclidean_to_angular_shape_metric]},
    {inp: "procrustes-angular", out: "procrustes-euclidean", function: [#angular_to_euclidean_shape_metric]},

    // {inp: "procrustes-angular", out: "procrustes-score", function: []}

    // TODO: use recursion? (computer compositions here instead of backends? => create list of transforms + derived transforms)
    {inp: "procrustes-sq-euclidean", out: "procrustes-angular", function: [#sqrt, #euclidean_to_angular_shape_metric]},
    // TODO: raise error
    // {inp: "procrustes-angular", out: "procrustes-sq-euclidean", function: [#angular_to_euclidean_shape_metric, #square]},

    // manually write the relevant compositions for now
    // default procrustes = procrustes-angular
    {inp: "procrustes-sq-euclidean", out: "procrustes", function: [#sqrt, #euclidean_to_angular_shape_metric]},
    {inp: "procrustes-angular", out: "procrustes", function: []},

    // TODO: procrustes-score
    {inp: "procrustes-angular", out: "procrustes-score", function: [#angular_dist_to_score]},

    // Identity tsf for same metrics that have different name
    // TODO: two-way transforms
    {inp: "shape_metric-angular-alpha10e-1", out: "procrustes-angular", function: []},
    {inp: "procrustes-angular", out: "shape_metric-angular-alpha10e-1", function: []},
    // composition written manually to identify "procrustes-angular" and "shape_metric-angular-alpha10e-1"
    {inp: "procrustes-sq-euclidean", out: "shape_metric-angular-alpha10e-1", function: [#sqrt, #euclidean_to_angular_shape_metric]},
    // procrustes-sq-euclidean => procrustes-sq = shape_metric-euclidean-alpha10e-1
    {inp: "procrustes-sq-euclidean", out: "shape_metric-euclidean-alpha10e-1", function: [#sqrt]},

    // cca => cca-angular = shape_metric-angular-alpha0e-1
    {inp: "cca", out: "shape_metric-angular-alpha0e-1", function: [#arccos]},
    // cca => cca-angular => cca-euclidean = shape_metric-euclidean-alpha0e-1
    {inp: "cca", out: "shape_metric-euclidean-alpha0e-1", function: [#arccos, #angular_to_euclidean_shape_metric]},
    

    {inp: "shape_metric-angular-alpha0e-1", out: "cca-angular", function: []},
    // {out: "shape_metric-angular-alpha1", inp: "procrustes-angular"},
    // {out: "shape_metric-angular-alpha0", inp: "cca-angular"},

    // Duality of Bures and Shape Distances with Implications for Comparing Neural Representations
    {inp: "nbs", out: "procrustes-angular", function: [#arccos]},
    {inp: "procrustes-angular", out: "nbs", function: [#cos]},
    {inp: "bures_distance", out:"procrustes-euclidean", function: []},
    {inp: "procrustes-euclidean", out:"bures_distance", function: []},
]
// transforms: [...{inp: #MetricName, out: #MetricName, function?: [...]}]
