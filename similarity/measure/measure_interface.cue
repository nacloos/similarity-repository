package measure
import(
    "list"
    "github.com/similarity/utils"
)
#target: utils.#target
#Seq: utils.#Seq
#ModKeys: utils.#ModKeys


// config for the python implementation of the Measure class
#Measure: self={
    // path to measure class
    // TODO: #path is not defined in generated schemas (e.g. netrep Linearmeasure)
    if self["_target_"] != _|_ {
        _target: self["_target_"]
    }
    if self["_target_"] == _|_ {
        _target: #path
    }

    #path: string
    // set to true if refer to a function instead of a class
    #partial: bool | *false
    // attribute name of the method to call on the measure class
    #call_key: string | null | *"fit_score"

    #function: bool | *false
    if #function {
        #partial: true
        #call_key: null
    }

    // preprocessing steps to apply on X, Y before the measure
    #preprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: #ModKeys | *["X", "Y"]
        #out_keys: #ModKeys | *["X", "Y"]
        ...
    }]
    // postprocessing steps to apply on the measure score (e.g. normalize to [0, 1])
    #postprocessing: [...{
        #path: string
        #partial: bool | *true
        #in_keys: #ModKeys | *["score"]
        #out_keys: #ModKeys | *["score"]
        ...
    }]
    "_preprocessing_": #preprocessing
    "_postprocessing_": #postprocessing
    // TODO: allow appending steps to user defined postprocessing
    #_postprocessing: [...]

    if len(#_postprocessing) > 0 {
        _postprocessing: #postprocessing + #_postprocessing
    }
    if len(#_postprocessing) == 0 {
        _postprocessing: #postprocessing
    }
    
    // fit_score interface
    // #fit_score_inputs: [string, string] | *["X", "Y"]
    #fit_score_inputs: #ModKeys | *["X", "Y"]
    // TODO: problem with output keys [["mean", 0], "score"] if use #ModKeys
    // #fit_score_outputs: #ModKeys | *["score"]
    #fit_score_outputs: [...] | *["score"]
    
    // TODO: allow it to be overwritten?
    if self.#call_key == null {
        // don't need to pass "self" because measure is already a function
        // #fit_score_in_keys: [
        //     ["X", self.#fit_score_inputs[0]],
        //     ["Y", self.#fit_score_inputs[1]]
        // ]
        #fit_score_in_keys: self.#fit_score_inputs
    }
    if self.#call_key != null {
        // need to pass "self" because target is a class method
        // #fit_score_in_keys: [
        //     ["measure", "self"], 
        //     ["X", self.#fit_score_inputs[0]], 
        //     ["Y", self.#fit_score_inputs[1]]
        // ]
        #fit_score_in_keys: [
            ["measure", "self"], 
            self.#fit_score_inputs[0], 
            self.#fit_score_inputs[1]
        ]
    }


    // TODO: backend
    // TODO: interface (need special_kwargs: ["interface", "postprocessing", "preprocessing"])
    // interface: {
    //     fit_score: ["__call__", "fit_score"]
    //     // fit_score: ["score", "__call__"]
    // }
    // #convert_to_distance: { ... } | *null
    // distance: bool | *false
    // if distance { #convert_to_distance != null}

    // constructor kwargs
    ...


    // definitions not working with transforms (when copy all the fields of a measure)
    "_fit_score_in_keys_": self.#fit_score_in_keys
    "_fit_score_outputs_": #fit_score_outputs
    "_call_key_": #call_key

    #reserved_keywords: ["_out_", "_preprocessing_", "_postprocessing_", "_call_key_", "_fit_score_in_keys_", "_fit_score_outputs_"]
    // pipeline to create the measure object
    "_out_": #target & {
        #path: "similarity.measure.Measure"
        
        measure: #target & {
            // set path and kwargs for measure
            #path: self.#path
            #partial: self.#partial
            // loop through the keys in self (automatically ignores keys starting with _ or #)
            // keys start and ending with _ are reserved keywords
            // for k, v in self if k != "_out_" {
            for k, v in self 
            if !list.Contains(#reserved_keywords, k) {
                (k): v
            }
        }
        
        fit_score: #Seq & {#modules: [
            // preprocessing steps
            for p in #preprocessing {
            // for p in self["_preprocessing_"] {
                // TODO: need to make it more general?
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            },
            // call measure
            #target & {
                // #call_key can be used to specify a method to call on the measure class

                // #in_keys: self.#fit_score_in_keys 
                #in_keys: self["_fit_score_in_keys_"] 
                if self["_call_key_"] == null {
                    "_target_": self._target
                    // #path: measure.#path
                    // #in_keys: ["X", "Y"]
                    measure
                }
                if self["_call_key_"] != null {
                    "_target_": "\(self._target).\(self.#call_key)"
                    // #path: "\(measure.#path).\(self.#call_key)"
                    // #in_keys: [["measure", "self"], "X", "Y"]
                }
                // use partial because target is a function here
                #partial: true
                #out_keys: self["_fit_score_outputs_"]
            },
            // postprocessing steps
            // for p in #postprocessing {
            // TODO
            for p in self._postprocessing {
            // for p in self["_postprocessing_"] {
                #target & {
                    #path: p.#path
                    #partial: p.#partial
                    #in_keys: p.#in_keys
                    #out_keys: p.#out_keys
                }
            }
        ]
        #in_keys: ["measure", "X", "Y"]
        #out_keys: [["score", null]]  // return the score value as a number (not a dict)
        }
    }
}
