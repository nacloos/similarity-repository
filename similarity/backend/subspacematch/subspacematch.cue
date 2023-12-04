package backend

github: "https://github.com/MeckyWu/subspace-match"

measure: {
    max_match: #Measure & {
        #path: "similarity.backend.subspacematch.score.maximum_match_score"
        #preprocessing: [
            #reshape2d
        ]
        #function: true
        // TODO: vary param
        epsilon: 0.25
    }
}

