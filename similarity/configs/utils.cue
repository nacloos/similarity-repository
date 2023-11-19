package similarity


#ModKeys: [...string] | {[string]: string}
#target: {
    #path: string
    #partial: bool | *false
    #recursive: bool | *true
    #convert: *null | "object" | "all" | "partial"

    // list of keys or dict of key mappings (e.g. {"a": "b"} corresponds to 'target(a=data["b"])')
    // if in_keys and out_keys are given, use DictModule
    #in_keys: #ModKeys | *null
    #out_keys: #ModKeys | *null
    #call_attr?: string
   
    #args: [...] | *[]

    "_target_": #path
    "_partial_": #partial
    "_recursive_": #recursive
    "_convert_": #convert
    "_args_": #args
    "_out_"?: _
    "_wrap_"?: {...}
    
    ...

    // TODO: "_in_keys_", "_out_keys_" => if given, wrap target with DictModule
    // so that don't need the extra level with the module key
    if #in_keys != null || #out_keys != null {
        "_wrap_": {
            dict_mod: {
                "_target_": "config_utils.dict_module.DictModule"
                in_keys: #in_keys
                out_keys: #out_keys
                if #call_attr != _|_ { call_attr: #call_attr }
            }
            ...
        }
    }    
}

#sample: {
    #target & {
        #path: "generative_task.modular_dynamics.ConfigSampler"
        // don't want to resolve the kwargs before calling ConfigSampler
        "_recursive_": false
        "_convert_": null  
    }
    #keys: [...string]
    // use dict for keys to allow overwriting
    "_sample_": keys: {
        for key in #keys {
            "\(key)": null
        }
    }
    ...
}

// variable resolved at runtime
#ref: {
    #key: string
    value: "${v:\(#key)}"
}

#tuple: {
	"_target_": "builtins.tuple"
	"_args_": [#args]
	#args: [..._]
}
#set: {
	"_target_": "builtins.set"
	"_args_": [#args]
	#args: [..._]
}


// TODO: use target instead?
#Mod: {
    #target & {#path: "config_utils.dict_module.DictModule"}
    module: {
        ...
    }
    // TODO: allow typing of in_keys and out_keys
    in_keys: [...string]
    out_keys: [...string]
}


#Seq: #target & {
    #path: "config_utils.dict_module.DictSequential"
    #modules: [...{
        // can't be null
        #in_keys: #ModKeys
        #out_keys: #ModKeys | *null
        ...
    }]
    "_args_": #modules
}
