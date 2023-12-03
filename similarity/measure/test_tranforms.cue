package test_transforms


metric: [string]: {
    #postprocessing: [...]
    "_postprocessing_": #postprocessing
}
metric: {
    a: {
        // value: 10
        #postprocessing: []
    }
    b: {
        // value: 20
        #postprocessing: []
    }
    "a-sq": {
        #postprocessing: []
    }
}


_transforms: [
    // acyclic?
    {inp: "a", out: "a-sq", function: ["square"]},
    // {inp: "a", out: "a-sq", function: ["square"], inverse: ["sqrt"]},

    // {inp: "a-sq", out: "a", function: ["sqrt"], inverse: ["square"]},
    {inp: "a", out: "a-angular", function: ["angular"]},

    // {inp: "a-sq", out: "a-sq-angular", function: ["angular"], inverse: ["euclidean"]},
    // {inp: "a-sq", out: "a-sq-angular", function: ["angular"]},

    // {inp: "a", out: "a-sq-angular", function: ["square", "angular"]},  // must be the same as the composition
    
    // TODO: two ways to come to the same tsf?
    // {inp: "b", out: "a-sq-angular", function: ["square", "angular"]},
    // {inp: "a-sq-angular", out: "a-sq-angular-score", function: ["score"]}, 

    // {inp: "a-sq-angular", out: "a", function: ["sqrt", "angular"]},
]


// have to guarantee that there is no more than one edge between each pair of nodes
#compose_transforms: {
    // out: for out_k, v in transforms_by_out {
    //     for inp_k, tsf in v {

    //     }
    // }
    transforms: [out=string]: [inp=string]: { ... }
    i: int | *0
    max_depht: int | *3
    // transforms_by_out: {for T in transforms {(T.out): (T.inp): T}}
    if i >= max_depht {
        out: transforms
    }
    if i < max_depht {
        // out: [for v in (#compose_transforms & {"transforms": self.transforms, "i": i+1}).out {
        //     a: 10
        // }]
        // #composed_transforms: (#compose_transforms & {"transforms": self.transforms, "i": i+1}).out
        // _transforms_by_inp: {for T in _composed_transforms {(T.inp): (T.out): T}}
        // #transforms_by_out: {for T in #composed_transforms {(T.out): (T.inp): T}}

        out: {
            for out1, v in (#compose_transforms & {"transforms": transforms, "i": i+1}).out {
                (out1): v
                for inp1, T1 in v  // for all edges inp1 -> out1 
                if transforms[inp1] != _|_  // where inp1 is an output of another edge
                for inp2, T2 in transforms[inp1]  // for all edges inp2 -> inp1
                {
                    (out1): (inp2): {
                        inp: T2.inp
                        out: T1.out
                        function: T2.function + T1.function
                    }
                    // inverse of the composition
                    if T1["inverse"] != _|_ && T2["inverse"] != _|_ {
                        (inp2): (out1): {
                            inp: T1.out
                            out: T2.inp
                            function: T1.inverse + T2.inverse
                        }
                    }
                }
                // if transforms[T1.inp] != _|_ {
                //     (out1): (transforms[T1.inp]): {
                //         inp: transforms[inp1].inp
                //         out: T1.out
                //         function: transforms[inp1].function + T1.function
                //     }
                // } 
            }
        }

        // T2: transforms[0]
        // out: [
        //     // compose pairs (T1, T2) s.t. T1.out == T2.inp and T2.out not already in transforms
        //     for T1 in (#compose_transforms & {"transforms": self.transforms, "i": i+1}).out if  _transforms_by_inp[T1.out] != _|_
        //     for T2 in _transforms_by_inp[T1.out] 
        //     if T1.out == T2.inp
        //     if transforms[T2.out] == _|_ {
        //         inp: T1.inp, out: T2.out, function: T1.function + T2.function
        //     }
        // ]
    }
}


#derive_metrics: {
    metrics: [string]: {...}
    transforms: [...{inp: string, out: string, function: _}]
    max_depht: int | *3
    transforms_by_out: {for T in _transforms {(T.out): (T.inp): T}}
    all_transforms:(#compose_transforms & {transforms: transforms_by_out, "max_depht": max_depht}).out

    derived_metrics: {
        for out, v in all_transforms if metrics[out] == _|_  // if the tsf output isn't already a metric
        for inp, T in v if metrics[inp] != _|_  {  // if the tsf input is a metric
            // (out): metrics[inp]
            (out): {
                // all keys except postprocessing
                for k, v in metrics[inp] if k != "_postprocessing_" {
                    (k): v
                }
                // append tsf to postprocessing
                "_postprocessing_": metrics[inp]["_postprocessing_"] + T.function
            }
        }
    }
    out: {
        metrics
        derived_metrics
    }
}

// y: (#compose_transforms & {transforms: _transforms}).out
// y: (#compose_transforms & {transforms: transforms_by_out}).out
y: (#derive_metrics & {transforms: _transforms, metrics: metric}).out
tsfs: (#derive_metrics & {transforms: _transforms, metrics: metric}).all_transforms



// metric2: {
//     for k, v in metric
//     for tsf in transforms
//     if tsf.inp == k {
//         (tsf.out): 10
//     }
//     ...
// }


// metrics: {a: 5}
// for k, v in {a: 5}
//  (a-angular): v
//  (a-sq): v
// out: {a: 5a-angular: 5, a-sq: 5}

// {a: 5, a-anguar: 5, a-sq: 5}
// for k, v in {a: 5, a-anguar: 5, a-sq: 5}



// very slow and not working (conflict error postprecessing list of different len)
#derive_metrics2: {
    i: int | *0
    max_depht: int | *3  // max depth for the recursion (i.e. max number of transform compositions)
    metrics: [string]: {...}
    transforms: [...{inp: string, out: string, function: _}]

    out: {
        // copy existing metrics
        metrics
        // recursively add derived metrics
        if i < max_depht {
            // for each metric, check if there is a transform
            for k, v in (#derive_metrics & {"metrics": metrics, "transforms": transforms, "i": i+1}).out
            for tsf in transforms
            if tsf.inp == k && metrics[tsf.out] == _|_ {
                // add the derived metric
                // (tsf.out): out[k]
                (tsf.out): {
                    // all keys except postprocessing
                    for kk, vv in out[k] if kk != "_postprocessing_" {
                        (kk): vv
                    }
                    // append tsf to postprocessing
                    "_postprocessing_": out[k]["_postprocessing_"] + tsf.function
                }
                // why these are not working?
                // (tsf.out): self.metrics[k]
                // (tsf.out): metrics[k]
            }
        }
    }
}

// x: (#derive_metrics & {metrics: metric, transforms: _transforms}).out

