package similarity

#reshape2d: {
    #path: "similarity_measures.processing.flatten_3d_to_2d"
    #partial: true
}
#pca: {
    #path: "similarity_measures.processing.pca_preprocessing"
    #partial: true
    n_components: number
}
#arccos: {
    #path: "similarity_measures.processing.angular_dist"
    #partial: true
    #in_keys: ["score"]
    #out_keys: ["score"]
}
#angular_dist_to_score: {
    // normalize distance [0, pi/2] to score [1, 0]
    #path: "similarity_measures.processing.angular_dist_to_score"
    #partial: true
    #in_keys: ["score"]
    #out_keys: ["score"]
}

#Metric: self={
    // path to metric class
    #path: string
    // set to true if refer to a function instead of a class
    #partial: bool | *false
    // attribute name of the method to call on the metric class
    #call_key: string | null | *"fit_score"
    // preprocessing steps to apply on X, Y before the metric
    #preprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: ["X", "Y"]
        #out_keys: ["X", "Y"]
        ...
    }]
    // postprocessing steps to apply on the metric score (e.g. normalize to [0, 1])
    #postprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: ["score"]
        #out_keys: ["score"]
        ...
    }]
    // constructor kwargs
    ...

    // pipeline to create the metric object
    "_out_": #target & {
        #path: "similarity_measures.metric.Metric"
        
        metric: #target & {
            // set path and kwargs for metric
            #path: self.#path
            #partial: self.#partial
            // loop through the keys in self (automatically ignores keys starting with _ or #)
            { for k, v in self if k != "_out_" { (k): v } }
        }
        
        fit_score: #Seq & {#modules: [
            // preprocessing steps
            for p in #preprocessing {
                // TODO: need to make it more general?
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            },
            // call metric
            #target & {
                // #call_key can be used to specify a method to call on the metric class
                if self.#call_key == null {
                    #path: metric.#path
                    // don't need to pass "self" because metric is already a function
                    #in_keys: {"X": "X", "Y": "Y"}
                }
                if self.#call_key != null {
                    #path: "\(metric.#path).\(self.#call_key)"
                    // need to pass "self" because target is a class method
                    #in_keys: {"self": "metric", "X": "X", "Y": "Y"}
                }
                // use partial because target is a function here
                #partial: true
                // TODO: okay to use generic term "score" even though the output might be a distance?
                #out_keys: ["score"]
            },
            // postprocessing steps
            for p in #postprocessing {
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            }
        ]}
    }
}

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
    #path: "similarity_measures.cka.linear_CKA"
    // don't use call_key because linear_CKA is already a function (not a class with a fit_score method)
    #call_key: null
    #partial: true
    #preprocessing: [#reshape2d]
    #postprocessing: [#arccos, #angular_dist_to_score]
}

