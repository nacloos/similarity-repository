package backend

// _methods: [
//     "stress",
//     "angular_cka",
//     "affine_invariant_riemannian",
//     "euclidean_shape_metric",
//     "angular_shape_metric"
// ]

#array_to_tensor: #target & {
    #path: "similarity.processing.array_to_tensor"
    #partial: true
}
#tensor_to_float: #target & {
    #path: "similarity.processing.tensor_to_float"
    #partial: true
}


measure: {
    [string]: #Measure & {
        #path: "repsim.compare"
        // TODO: don't work for cka-angular??
        // #function: true
        // #fit_score_inp_keys: [["X", "x"], ["Y", "y"]]
        #fit_score_in_keys: [["X", "x"], ["Y", "y"]] // why have to add this here?
        #function: true
        #preprocessing: [
            // repsim.compare requires torch tensors as inputs
            #array_to_tensor
        ]
        #postprocessing: [ #tensor_to_float ]
    }
    "cka-angular": {
        method: "angular_cka"
    }
    // TODO: TypeError: __init__() missing 1 required positional argument: 'p'
    "procrustes-angular": {
        // #path: "repsim.compare"
        // #fit_score_in_keys: [["X", "x"], ["Y", "y"]] // why have to add this here?
        // #function: true
        // #preprocessing: [
        //     // repsim.compare requires torch tensors as inputs
        //     #array_to_tensor
        // ]
        // #postprocessing: [ #tensor_to_float ]
        method: "angular_shape_metric"
        alpha: 1
        // TODO: vary this param?
        p: 100  // nb of components to keep (value used in the paper)
    }
    "procrustes-euclidean": {
        method: "euclidean_shape_metric"
        alpha: 1
        p: 100
    }
    "cca-angular": {
        method: "angular_shape_metric"
        alpha: 0
        p: 100
    }
    "cca-euclidean": {
        method: "euclidean_shape_metric"
        alpha: 0
        p: 100
    }
    "riemannian_metric": {
        method: "affine_invariant_riemannian"
    }

    // TODO
    // [string]: {
    //     #path: "repsim.compare"
    //     // TODO: don't need to use the term "key"
    //     #inputs: [
    //         {name: "x", type: #TorchTensor},
    //         {name: "y", type: #TorchTensor}
    //     ]
    //     #outputs: [
    //         {name: "score", type: #TorchTensor}
    //     ]
    //     #scoring_method: "inner_product" | "angle" | "cosine" | "distance" | "square_distance"
    //     method: string
    // }
    // for method in _methods {
    //     (method): {method: method}
    // }
}


// TODO
// class CompareType(enum.Enum):
//     """Comparison type for repsim.compare and repsim.pairwise.compare.

//     CompareType.INNER_PRODUCT: an inner product like x @ y.T. Large values = more similar.
//     CompareType.ANGLE: values are 'distances' in [0, pi/2]
//     CompareType.COSINE: values are cosine of ANGLE, i.e. inner-product of unit vectors
//     CompareType.DISTANCE: a distance, like ||x-y||. Small values = more similar.
//     CompareType.SQUARE_DISTANCE: squared distance.

//     Note that INNER_PRODUCT has a different sign than the others, indicating that high inner-product means low distance
//     and vice versa.
//     """

//     INNER_PRODUCT = -1
//     ANGLE = 0
//     COSINE = 1
//     DISTANCE = 2
//     SQUARE_DISTANCE = 3

card: {
    github: "https://github.com/wrongu/repsim/"
}