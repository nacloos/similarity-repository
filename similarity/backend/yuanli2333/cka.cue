package backend


metric: {
    cka: #Metric & {
        // TODO: directly refer to github file instead of having to copy it here
        #path: "similarity.backend.yuanli2333.cka.linear_CKA"
        // TODO: relative path? ".cka/linear_CKA"
        // #path: "similarity.metrics.backends.yuanli2333.cka.linear_CKA"
        // don't use call_key because linear_CKA is already a function (not a class with a fit_score method)
        #function: true
        // #call_key: null
        // #partial: true
        #preprocessing: [#reshape2d]
        #postprocessing: [#arccos, #angular_dist_to_score]    }
}
