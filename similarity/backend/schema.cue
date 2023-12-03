package backend
import(
    // can't import measure here because it will cause a circular dependency
    // => move schemas to similarity package...
    "github.com/similarity"
    "github.com/similarity/utils"
    "github.com/similarity/processing"
    // TODO: if user different card package in measure, then cannot just do similarity.make(package="measure", ...)
    measure_cards "github.com/similarity/measure:card"
)
#target: utils.#target
#measure: similarity.#measure
// #MeasureName: similarity.#MeasureName

// TODO: get measure names from cards (but can't import measure here because of cyclic import)
// #measure_names: [
//     "procrustes", 
//     "cca", 
//     "cca_mean_sq_corr",
//     "pwcca",
//     "svcca", 
//     "cka", 
//     "rsa", 
//     "linear_regression",
//     "pls",
//     "permutation",
//     "imd",
//     "max_match"
// ]
#measure_names: [for k, _ in measure_cards.cards { k }]

#MeasureName: or(#measure_names)

#Card: {
    id?: string,  // TODO: automatically extract id?
    name?: string
    github?: string
    website?: string
    citation?: string | [...string]
}
card: #Card


// close restrict the keys of measure to be in #MeasureName
measure: close({
    // #measure doesn't do anything by default, it just adds functionalities
    // [#MeasureName]: #measure  // why is it so slow???
    [#MeasureName]: { ... }
})


// for k, v in measure {
//     measure: (k): #measure & v
// }

// prevode helper functions for pre and post processing
#reshape2d: processing.#reshape2d
#arccos: processing.#arccos
#angular_dist_to_score: processing.#angular_dist_to_score
#pca: processing.#pca


// #TensorTensor: {}

// #Float: #target & {
//     #path: "jaxtyping.Float"
// }
// #Array: #target & {
//     #path: "jaxtyping.Array"
//     // jaxtyping
//     dtype: _ | *#Float
//     shape: [...string]

//     "_out_": {
//         // TODO: construct jaxtyping type
//         type: "\(dtype)[Array, \(shape)]"
//     }
// }
// #TorchTensor: #target & {
//     #path: "torch.Tensor"
// }
// #NumpyArray: #target & {
//     #path: "numpy.ndarray"
// }
