package processing



#reshape2d: {
    #path: "similarity.processing.flatten_3d_to_2d"
    #partial: true
}
#pca: {
    #path: "similarity.processing.pca_preprocessing"
    #partial: true
    n_components: number
}
#arccos: {
    #path: "similarity.processing.angular_dist"
    #partial: true
    #in_keys: ["score"]
    #out_keys: ["score"]
}
#angular_dist_to_score: {
    // normalize distance [0, pi/2] to score [1, 0]
    #path: "similarity.processing.angular_dist_to_score"
    #partial: true
    #in_keys: ["score"]
    #out_keys: ["score"]
}
