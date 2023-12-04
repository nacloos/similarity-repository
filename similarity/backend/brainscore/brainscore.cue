package backend

github: "https://github.com/brain-score/brain-score"


// TODO: temporarily copied brainscore package in this repo (had issues installing it as a package because of old version of sklearn)
_brainscore_path: "brainscore"  

    
#numpy_to_brainio: #target & {
    #path: "similarity.backend.brainscore.utils.numpy_to_brainio"
    #partial: true
}
#aggregate_score: #target & {
    #path: "similarity.backend.brainscore.utils.aggregate_score"
    #partial: true
}


measure: {
    [string]: #Measure & {
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
    linear_regression:  {
        // #path: "similarity.backend.brainscore.utils.pls_metric"
        #path: "\(_brainscore_path).metrics.regression.CrossRegressedCorrelation"
        regression: null
        correlation: #target & {
            #path: "\(_brainscore_path).metrics.regression.pearsonr_correlation"
        }
        #fit_score_inputs: [["X", "source"], ["Y", "target"]]
    }
    cka: {
        #path: "\(_brainscore_path).metrics.cka.CKAMetric"
        #fit_score_inputs: [["X", "assembly1"], ["Y", "assembly2"]]
    }
    // inferred from the code
    "rsa-correlation-spearman": {
        #path: "\(_brainscore_path).metrics.rdm.RDMMetric"
        #fit_score_inputs: [["X", "assembly1"], ["Y", "assembly2"]]
    }
    correlation: {
        #path: "\(_brainscore_path).metrics.correlation.Correlation"
        #fit_score_inputs: [["X", "source"], ["Y", "target"]]
    }
}
