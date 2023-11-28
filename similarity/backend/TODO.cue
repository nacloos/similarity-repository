

// TODO: make("...", strict=True) => don't do any type convertion or reshaping, raise error if type doesn't comply

metric: {

    test: {
        #inputs: [
            {name: "rep1", type: #Array & {shape: ["batch", "neuron"]}},
            {name: "rep2", type: #Array & {shape: ["batch", "neuron"]}},
        ]
        // TODO?
        #inputs2: {
            rep1: #Array & {shape: ["batch", "neuron"]},
            rep2: #Array & {shape: ["batch", "neuron"]},
        }
    }

}