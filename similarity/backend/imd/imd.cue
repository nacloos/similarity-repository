package backend
import(
    "github.com/similarity/papers"
)
github: "https://github.com/xgfs/imd"
paper: [papers.tsitsulin2020]

measure: {
    imd: #Measure & {
        #path: "msid.msid_score"
        #function: true
        #preprocessing: [
            #reshape2d
        ]
        #fit_score_inputs: [["X", "x"], ["Y", "y"]]
    }
}

