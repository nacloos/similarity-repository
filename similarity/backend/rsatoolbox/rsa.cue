package backend


#rdm_method: string
#compare_method: 
    "cosine" |
    "spearman" |
    "corr" |
    "kendall" |
    "tau-b" |
    "tau-a" |
    "rho-a" |
    "corr_cov" |
    "cosine_cov" |
    "neg_riem_dist"


// TODO: all the variations of rsa?
metric: {
    #base_rsa: #Metric & {
        #path: "similarity.backend.rsatoolbox.rsa.compute_rsa"
        #call_key: null
        #partial: true
        #preprocessing: [#reshape2d]
        #postprocessing: [#arccos, #angular_dist_to_score]
        rdm_method: string
        compare_method: string
    }

    rsa: #base_rsa & {
        rdm_method: #rdm_method | *"euclidean"
        compare_method: #compare_method | *"cosine"
    }
}
