package backend
import(
    // "github.com/similarity/processing"
    "github.com/similarity/measure:card"
)

github: "https://github.com/js-d/sim_metric"
paper: [card.papers.ding2021]


// copied sim_metrics/dists in this folder
_measure_path: "similarity.backend.sim_metric.dists.scoring"
_utils_path: "similarity.backend.sim_metric.utils"

#transpose: #target & {
    #path: "similarity.processing.transpose"
    #partial: true
}

measure: {
    [string]: #Measure & {
        #preprocessing: [
            #reshape2d,
            // sim_metric scoring functions expect representations to be in shape (neuron, sample)
            // but similarity.measure expects (sample, neuron)
            // processing.#transpose
            #transpose
        ]
    }
    cca: {
        #path: "\(_utils_path).mean_cca_corr"
        #function: true
        #fit_score_inputs: [["X", "rep1"], ["Y", "rep2"]]
    // TODO
    // cca_mean_corr
    }
    cca_mean_sq_corr: {
        #path: "\(_utils_path).mean_sq_cca_corr"
        #function: true
        #fit_score_inputs: [["X", "rep1"], ["Y", "rep2"]]
    }
    pwcca: {
        #path: "\(_utils_path).pwcca_dist"
        #function: true
        #fit_score_inputs: [["X", "rep1"], ["Y", "rep2"]]
    }
    cka: {
        #path: "\(_measure_path).lin_cka_dist"
        // TODO: what is cka prime?
        // #path: "\(_measure_path).lin_cka_prime_dist"
        #function: true
        #fit_score_inputs: [["X", "A"], ["Y", "B"]]
        // TODO: (Ding, 2021): d_CKA = 1 - CKA
        // this is not a proper measure?
        #postprocessing: [
            #target & {#path: "similarity.processing.one_minus_score", #partial: true}
        ]
    }
    "procrustes-sq-euclidean": {
        // TODO: different from netrep?
        #path: "\(_measure_path).procrustes"
        #function: true
        #fit_score_inputs: [["X", "A"], ["Y", "B"]]
    }
}

