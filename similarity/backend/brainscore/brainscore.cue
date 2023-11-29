package backend

_brainscore_path: "brainscore"  // TODO: problem installing brainscore as a package
// _brainscore_path: "similarity.backend.brainscore.brainscore"

    
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
        #path: "\(_brainscore_path).metrics.regression.CrossRegressedCorrelation"
        regression: null
        correlation: #target & {
            #path: "\(_brainscore_path).metrics.regression.pearsonr_correlation"
        }
        #fit_score_inputs: [["X", "source"], ["Y", "target"]]
    }
    cka: #Metric & {
        #path: "\(_brainscore_path).metrics.cka.CKAMetric"
        #fit_score_inputs: [["X", "assembly1"], ["Y", "assembly2"]]
    }
    // inferred from the code
    // "rsa-correlation-spearman": {...}
}


card: {
    id: "brainscore"
    name: "Brain-Score"
    github: "https://github.com/brain-score/brain-score"
    website: "www.brain-score.org"
    citation: [
"""
@article{SchrimpfKubilius2018BrainScore,
    title={Brain-Score: Which Artificial Neural Network for Object Recognition is most Brain-Like?},
    author={Martin Schrimpf and Jonas Kubilius and Ha Hong and Najib J. Majaj and Rishi Rajalingham and Elias B. Issa and Kohitij Kar and Pouya Bashivan and Jonathan Prescott-Roy and Franziska Geiger and Kailyn Schmidt and Daniel L. K. Yamins and James J. DiCarlo},
    journal={bioRxiv preprint},
    year={2018},
    url={https://www.biorxiv.org/content/10.1101/407007v2}
}
""",
"""
@article{Schrimpf2020integrative,
    title={Integrative Benchmarking to Advance Neurally Mechanistic Models of Human Intelligence},
    author={Schrimpf, Martin and Kubilius, Jonas and Lee, Michael J and Murty, N Apurva Ratan and Ajemian, Robert and DiCarlo, James J},
    journal={Neuron},
    year={2020},
    url={https://www.cell.com/neuron/fulltext/S0896-6273(20)30605-X}
}
"""
    ]       
}


