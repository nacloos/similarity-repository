package backend

metric: {
    max_match: #Metric & {
        #path: "similarity.backend.subspacematch.score.maximum_match_score"
        #preprocessing: [
            #reshape2d
        ]
        #function: true
        // TODO: how to evaluate metric with parameters?
        epsilon: 0.25
    }
}

card: {
    github: "https://github.com/MeckyWu/subspace-match"
}