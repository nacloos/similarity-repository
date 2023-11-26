package backend

metric: {
    imd: #Metric & {
        #path: "msid.msid_score"
        #function: true
        #preprocessing: [
            #reshape2d
        ]
        #fit_score_inputs: [["X", "x"], ["Y", "y"]]
    }
}

card: {
    github: "https://github.com/xgfs/imd"
}