package backend
import(
    // "github.com/similarity/processing"
)

// _metric_path: "similarity.backend.sim_metric.sim_metric.dists.scoring"
_metric_path: "similarity.backend.sim_metric.dists.scoring"
_utils_path: "similarity.backend.sim_metric.utils"

#transpose: #target & {
    #path: "similarity.processing.transpose"
    #partial: true
}

metric: {
    [string]: #Metric & {
        #preprocessing: [
            #reshape2d,
            // sim_metric scoring functions expect representations to be in shape (neuron, sample)
            // but similarity.Metric expects (sample, neuron)
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
        #path: "\(_metric_path).lin_cka_dist"
        // TODO: what is cka prime?
        // #path: "\(_metric_path).lin_cka_prime_dist"
        #function: true
        #fit_score_inputs: [["X", "A"], ["Y", "B"]]
    }
    procrustes: {
        // TODO: different from netrep?
        #path: "\(_metric_path).procrustes"
        #function: true
        #fit_score_inputs: [["X", "A"], ["Y", "B"]]
    }
}


card: {
    github: "https://github.com/js-d/sim_metric"
}