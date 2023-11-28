// package backend

_methods: [
    "stress",
    "angular_cka",
    "affine_invariant_riemannian",
    "euclidean_shape_metric",
    "angular_shape_metric"
]

metric: {
    [string]: {
        #path: "repsim.compare"
        // TODO: don't need to use the term "key"
        #inputs: [
            {name: "x", type: #TorchTensor},
            {name: "y", type: #TorchTensor}
        ]
        #outputs: [
            {name: "score", type: #TorchTensor}
        ]
        #scoring_method: "inner_product" | "angle" | "cosine" | "distance" | "square_distance"
        method: string
    }
    for method in _methods {
        (method): {method: method}
    }
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