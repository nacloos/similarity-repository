package similarity
import(
    "github.com/similarity/utils"
)
#target: utils.#target
#Seq: utils.#Seq
#ModKeys: utils.#ModKeys


// can't put these schemas in metric because causes circular dependency between backend and meetric packages
// backend needs to access the schemas and metric needs to access the backend implementations

// TODO: force each metric to have a card

// TODO: grob the folders in backend?
// TODO: backend card
// #BackendName: 
//     "netrep" | 
//     "rsatoolbox" |
//     "yuanli2333"


// #MetricBackend: {
//     [#MetricName]: #Metric
// }

#Metric: self={
    // path to metric class

    // TODO: #path is not defined in generated schemas (e.g. netrep LinearMetric)
    if self["_target_"] != _|_ {
        _target: self["_target_"]
    }
    if self["_target_"] == _|_ {
        _target: #path
    }

    #path: string
    // set to true if refer to a function instead of a class
    #partial: bool | *false
    // attribute name of the method to call on the metric class
    #call_key: string | null | *"fit_score"

    #function: bool | *false
    if #function {
        #partial: true
        #call_key: null
    }

    // preprocessing steps to apply on X, Y before the metric
    #preprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: ["X", "Y"]
        #out_keys: ["X", "Y"]
        ...
    }]
    // postprocessing steps to apply on the metric score (e.g. normalize to [0, 1])
    #postprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: ["score"]
        #out_keys: ["score"]
        ...
    }]

    // fit_score interface
    #fit_score_inputs: [string, string] | *["X", "Y"]

    // TODO: allow it to be overwritten?
    if self.#call_key == null {
        // don't need to pass "self" because metric is already a function
        #fit_score_in_keys: [
            ["X", self.#fit_score_inputs[0]],
            ["Y", self.#fit_score_inputs[1]]
        ]
    }
    if self.#call_key != null {
        // need to pass "self" because target is a class method
        #fit_score_in_keys: [
            ["metric", "self"], 
            ["X", self.#fit_score_inputs[0]], 
            ["Y", self.#fit_score_inputs[1]]
        ]
    }


    // TODO: backend

    // constructor kwargs
    ...

    // pipeline to create the metric object
    "_out_": #target & {
        #path: "similarity.metric.Metric"
        
        metric: #target & {
            // set path and kwargs for metric
            #path: self.#path
            #partial: self.#partial
            // loop through the keys in self (automatically ignores keys starting with _ or #)
            { for k, v in self if k != "_out_" { (k): v } }
        }
        
        fit_score: #Seq & {#modules: [
            // preprocessing steps
            for p in #preprocessing {
                // TODO: need to make it more general?
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            },
            // call metric
            #target & {
                // #call_key can be used to specify a method to call on the metric class

                #in_keys: self.#fit_score_in_keys 
                if self.#call_key == null {
                    "_target_": self._target
                    // #path: metric.#path
                    // #in_keys: ["X", "Y"]
                    metric
                }
                if self.#call_key != null {
                    "_target_": "\(self._target).\(self.#call_key)"
                    // #path: "\(metric.#path).\(self.#call_key)"
                    // #in_keys: [["metric", "self"], "X", "Y"]
                }
                // use partial because target is a function here
                #partial: true
                // TODO: okay to use generic term "score" even though the output might be a distance?
                #out_keys: ["score"]
            },
            // postprocessing steps
            for p in #postprocessing {
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            }
        ]
        #in_keys: ["metric", "X", "Y"]
        #out_keys: [["score", null]]  // return the score value as a number (not a dict)
        }
    }
}
