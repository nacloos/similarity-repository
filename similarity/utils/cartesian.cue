package utils
import (
    "list"
    "strings"
)


#Cartesian: self={
    inp: _
    _plist: [for _, v in self.inp {[for _, lv in v {[lv]}]}]
    _plist2: [
        for i, v in _plist {
            if i == 0 {v}
            if i > 0 {
                let prev = _plist2[i-1]
                [
                    for _, prevV in prev
                    for _, currV in v {list.Concat([prevV, currV])},
                ]
            }
        },
    ]
    out: {
        // index out of range if input is empty
        if len(_plist2) == 0 {[]}
        if len(_plist2) > 0 {_plist2[len(_plist2)-1]}        
    }
}
#names_from_params: self={
    inp: _
    out: [for p in (#Cartesian & {inp: self.inp}).out {
        {strings.Join(
            // convert param values to string (might not be necessary if constrain params to be strings)
            [for v in p { "\(v)"}], 
            "-"
        )}
    }]
}

#name_from_params: {
    base: string
    params: [...]
    out: base + "-" + strings.Join(params, "-")
}


params: {
	param1: [1, 2, 3]
	param2: ["a", "b"]
	param3: [-1, -2]
}
params2: {
    a: [1, 2, 3]
}

names: (#names_from_params & {inp: params}).out
for name in names {
    (name): "hey"
}
x: (#Cartesian & {inp: params}).out
y: (#Cartesian & {inp: params2}).out
