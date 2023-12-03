package backend
import(
    "github.com/similarity/measure"
    "github.com/similarity/utils"
    "github.com/similarity/processing"
)
// make these available in all the files in the child folders that belongs to the same package
#target: utils.#target
#Measure: measure.#Measure
#MeasureId: measure.#MeasureId


github?: string
paper?: [...]
citation?: string | [...string]

// close restricts the keys of measure to be in #MeasureId
// this prevents backends implementing measures that are not registered in measure
"measure": close({
    [#MeasureId]: { ... }
    // [#MeasureId]: #Measure  // very slow, add #Measure in each backend separately for now
})


// previde helper functions for pre and post processing
#reshape2d: processing.#reshape2d
#arccos: processing.#arccos
#angular_dist_to_score: processing.#angular_dist_to_score
#pca: processing.#pca

