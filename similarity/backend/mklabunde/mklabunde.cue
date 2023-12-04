package backend
import(
    "github.com/similarity/papers"
)

github: "https://github.com/mklabunde/survey_measures"
paper: [papers.klabunde2023]

measure: {
    [string]: #Measure
    "procrustes-sq-euclidean": {
        #path: "similarity.backend.mklabunde.procrustes"
        #function: true
    }
}
