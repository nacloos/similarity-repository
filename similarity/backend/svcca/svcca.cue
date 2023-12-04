package backend
import(
    "github.com/similarity/papers"
)
// python files in this folder were copied from the github repo (+ some small modifications)
github: "https://github.com/google/svcca/"
paper: [papers.raghu2017, papers.morcos2018]

measure: {
    [string]: #Measure & {
        #preprocessing: [
            #reshape2d,
            // svcca repo's functions expect data with shape (neuron, data_point)
            // and similarity.measure expects data with shape (data_point, neuron)
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
        #path: "similarity.backend.svcca.pwcca(modified).compute_pwcca"
        #function: true
        #fit_score_inputs: [["X", "acts1"], ["Y", "acts2"]]
        #fit_score_outputs: ["score", "_", "_"]  // use only the mean, which is the first output of compute_pwcca
    }
    pls: {
        #path: "similarity.backend.svcca.numpy_pls(modified).get_pls_similarity"
        #function: true
        #fit_score_inputs: [["X", "acts1"], ["Y", "acts2"]]
        // modified get_pls_similarity to return the mean of the eigenvalues, used as similarity score here
        #fit_score_outputs: [["mean", "score"]]
    }
}

