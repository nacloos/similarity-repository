package backend

measure: {
    max_match: #Measure & {
        #path: "similarity.backend.subspacematch.score.maximum_match_score"
        #preprocessing: [
            #reshape2d
        ]
        #function: true
        // TODO: how to evaluate measure with parameters?
        epsilon: 0.25
    }
}

card: {
    github: "https://github.com/MeckyWu/subspace-match"
}