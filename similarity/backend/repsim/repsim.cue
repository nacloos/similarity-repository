package backend

github: "https://github.com/wrongu/repsim/"


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
        #fit_score_in_keys: [["X", "x"], ["Y", "y"]]
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
    "procrustes-angular": {
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

}

// _methods: [
//     "stress",
//     "angular_cka",
//     "affine_invariant_riemannian",
//     "euclidean_shape_metric",
//     "angular_shape_metric"
// ]



