package backend


#rdm_method: string
#compare_method: string

// TODO: all the variations of rsa?
metric: {
    #base_rsa: #Metric & {
        #path: "similarity.backend.similarity_matrix.rsatoolbox.rsa.compute_rsa"
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
