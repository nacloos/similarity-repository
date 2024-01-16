package backend
import(
    "github.com/similarity/papers"
)

github: "https://github.com/amzn/xfer/blob/master/nn_similarity_index"
paper: [papers.tang2020]

measure: {
    [string]: #Measure & {
        #preprocessing: [
            #reshape2d,
            // compute kernel matrix according to eq (6) of the paper (https://arxiv.org/pdf/2003.11498.pdf)
            #target & {
                #path: "similarity.backend.nn_similarity_index.utils.compute_kernels"
                #partial: true
            }
        ]
        #fit_score_inputs: [["X", "kmat_1"], ["Y", "kmat_2"]]
        #function: true
    }
    // TODO: what do this correpond to?
    euclidean: #path: "similarity.backend.nn_similarity_index.utils.euclidean"
    cka: #path: "similarity.backend.nn_similarity_index.utils.cka"
    nbs: #path: "similarity.backend.nn_similarity_index.utils.nbs"
    // not in the original code but was simply implemented by rearranging the terms in SimIndex.nbs
    bures_distance: #path: "similarity.backend.nn_similarity_index.utils.bures_distance"
}