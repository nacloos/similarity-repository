package similarity

// TODO: delete?
#Permutation: {}
#Orthogonal: {}
#IsotropicScaling: {}
#InvertibleLinear: {}
#Translation: {}
#Affine: {}

#GroupTransformation: #Permutation | #Orthogonal | #IsotropicScaling | #InvertibleLinear | #Translation | #Affine

group_transformation: {
    // ref: Klabunde, 2023
    #PP: {name: "permutation"}
    #OT: {name: "orthogonal-transformation"}
    #IS: {name: "isotropic-scaling"}
    #ILT: {name: "invertible-linear-transformation"}
    #TR: {name: "translation"}
    #AT: {name: "affine-transformation"}    
}

preprocessing: {
    #CC: {name: "centered-columns"}
}
#Normalization: "mean-centering"
#AdjustingDimensionality: "pca" | "zero-padding"
#Flattening: {}
#Preprocessing: #Normalization | #Flattening | #AdjustingDimensionality


#SimilarityMeasure: {
    invariance: [...#GroupTransformation]
}



