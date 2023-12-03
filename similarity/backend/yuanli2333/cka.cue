package backend


measure: {
    cka: #Measure & {
        #path: "similarity.backend.yuanli2333.CKA.linear_CKA"
        // don't use call_key because linear_CKA is already a function (not a class with a fit_score method)
        #function: true
        // #call_key: null
        // #partial: true
        #preprocessing: [#reshape2d]
        // #postprocessing: [#arccos, #angular_dist_to_score] 
    }
}
