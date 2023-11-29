package backend


metric: {
    [string]: #Metric & {
        #preprocessing: [
            #reshape2d,
            // svcca repo's functions expect data with shape (neuron, data_point)
            // and similarity.Metric expects data with shape (data_point, neuron)
            #target & {
                #path: "similarity.processing.transpose"
                #partial: true
            }
        ]
    }
    cca: {
        #path: "similarity.backend.svcca.cca_core.get_cca_similarity"
        #function: true
        #fit_score_inputs: [["X", "acts1"], ["Y", "acts2"]]
        // get_cca_similarity returns a dict and value for "mean" is a tuple of len 2 with twice the same value
        #fit_score_outputs: [[["mean", 0], "score"]]
        verbose: false
    }
    pwcca: {
        #path: "similarity.backend.svcca.pwcca.compute_pwcca"
        #function: true
        #fit_score_inputs: [["X", "acts1"], ["Y", "acts2"]]
        #fit_score_outputs: ["score", "_", "_"]  // use only the mean, which is the first output of compute_pwcca
    }
    pls: {
        #path: "similarity.backend.svcca.numpy_pls.get_pls_similarity"
        #function: true
        #fit_score_inputs: [["X", "acts1"], ["Y", "acts2"]]
        // modify get_pls_similarity to return the mean of the eigenvalues, used as similarity score here
        #fit_score_outputs: [["mean", "score"]]
    }
}

card: {
    github: "https://github.com/google/svcca/"
    citation: [
"""
@incollection{NIPS2017_7188,
    title = {SVCCA: Singular Vector Canonical Correlation Analysis for Deep Learning Dynamics and Interpretability},
    author = {Raghu, Maithra and Gilmer, Justin and Yosinski, Jason and Sohl-Dickstein, Jascha},
    booktitle = {Advances in Neural Information Processing Systems 30},
    editor = {I. Guyon and U. V. Luxburg and S. Bengio and H. Wallach and R. Fergus and S. Vishwanathan and R. Garnett},
    pages = {6076--6085},
    year = {2017},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7188-svcca-singular-vector-canonical-correlation-analysis-for-deep-learning-dynamics-and-interpretability.pdf}
}
""",
"""
@incollection{NIPS2018_7815,
    title = {Insights on representational similarity in neural networks with canonical correlation},
    author = {Morcos, Ari and Raghu, Maithra and Bengio, Samy},
    booktitle = {Advances in Neural Information Processing Systems 31},
    editor = {S. Bengio and H. Wallach and H. Larochelle and K. Grauman and N. Cesa-Bianchi and R. Garnett},
    pages = {5732--5741},
    year = {2018},
    publisher = {Curran Associates, Inc.},
    url = {http://papers.nips.cc/paper/7815-insights-on-representational-similarity-in-neural-networks-with-canonical-correlation.pdf}
}
"""
    ]
}
