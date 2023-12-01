package test_transforms


metric: {
    a: 5
    // "a-sq": 10
    // "a-angular": 10
    ...
}


transforms: [
    {inp: "a", out: "a-sq"},
    {inp: "a", out: "a-angular"},
    // {inp: "a-sq", out: "a-sq-angular"},
    // {inp: "a-angular", out: "a-sq-angular"}
]


metric: self={
    for k, v in self
    for tsf in transforms
    if tsf.inp == k {
        (tsf.out): 10
    }
    ...
}


