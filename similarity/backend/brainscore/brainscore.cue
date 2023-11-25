package backend

#numpy_to_brainio: #target & {
    #path: "similarity.backend.brainscore.utils.numpy_to_brainio"
    #partial: true
}
#aggregate_score: #target & {
    #path: "similarity.backend.brainscore.utils.aggregate_score"
    #partial: true
}

// TODO: cross-validation?
metric: {
    [string]: {
        #preprocessing: [
            #reshape2d,
            #numpy_to_brainio
        ]
        #postprocessing: [
            #aggregate_score
        ]
        #call_key: "__call__"
    }
    // correlation: {...}
    // TODO: pls?
    linear_regression: #Metric & {
        // #path: "similarity.backend.brainscore.utils.pls_metric"
        #path: "brainscore.metrics.regression.CrossRegressedCorrelation"
        regression: null
        correlation: #target & {
            #path: "brainscore.metrics.regression.pearsonr_correlation"
        }
        #fit_score_inputs: [["X", "source"], ["Y", "target"]]
    }
    cka: #Metric & {
        #path: "brainscore.metrics.cka.CKAMetric"
        #fit_score_inputs: [["X", "assembly1"], ["Y", "assembly2"]]
    }
    // rsa: {...}
}




