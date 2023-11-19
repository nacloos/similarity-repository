package similarity


procrustes: #Metric & {
    #path: "netrep.metrics.LinearMetric"
    #preprocessing: [#reshape2d]
    #postprocessing: [#angular_dist_to_score]
    alpha: 1
}

cca: #Metric & {
    #path: "netrep.metrics.LinearMetric"
    #preprocessing: [#reshape2d]
    #postprocessing: [#angular_dist_to_score]
    alpha: 0
}

svcca: #Metric & {
    #path: "netrep.metrics.LinearMetric"
    #preprocessing: [#reshape2d, #pca & {n_components: 0.95}]
    #postprocessing: [#angular_dist_to_score]
    alpha: 0
}


cka: #Metric & {
    // TODO: directly refer to github file instead of having to copy it here
    #path: "similarity.cka.linear_CKA"
    // don't use call_key because linear_CKA is already a function (not a class with a fit_score method)
    #call_key: null
    #partial: true
    #preprocessing: [#reshape2d]
    #postprocessing: [#arccos, #angular_dist_to_score]
}

// TODO
// dsa: #Metric & {
//     #path: "netrep.metrics.LinearMetric"
//     #preprocessing: [#reshape2d]
//     #postprocessing: [#angular_dist_to_score]
//     alpha: 0
// }