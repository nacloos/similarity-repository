package property


#PropertyId: or([for k, v in property { k }])

property: [string]: {
    name?: string
    description?: string
}
property: {
    "permutation-invariant": {},
    "scale-invariant": {
        name: "Scale Invariant"
    }
    "rotation-invariant": {
        name: "Rotation Invariant"
    }
    "invertible-linear-invariant": {}
        "translation-invariant": {}
    "affine-invariant": {}
    score: {
        name: "Score"
        description: "Measure of similarity. A score of 1 indicates that the representations are identical."
    }
    metric: {
        name: "Metric"
        description: "Measure of distance."
    }
    // "riemannian-metric": {}
}

// measure: [#MeasureId]: {...}
measure: {
    // referece: (Klabunde, 2023) Table 1
    cca: properties: [
        "permutation-invariant",
        "scale-invariant",
        "rotation-invariant",
        "invertible-linear-invariant",
        "translation-invariant",
        "affine-invariant",
        "score",
    ]
    // problem with list is that the order matters, dict => can be overwritten more easily
    // cca: properties: {
    //     "permutation-invariant": true
    //     "scale-invariant": true
    //     "rotation-invariant": true
    //     "invertible-linear-invariant": true
    //     "translation-invariant": true
    //     "affine-invariant": true
    //     "score": true
    // }
    // svcca: properties: {
    //     "permutation-invariant": true
    //     "scale-invariant": true
    //     "rotation-invariant": true
    //     "translation-invariant": true
    //     "affine-invariant": false
    //     "score": true
    // }
    svcca: properties: [
        "permutation-invariant",
        "scale-invariant",
        "rotation-invariant",
        "translation-invariant",
        "score",
    ]
    pwcca: properties: [
        "scale-invariant",
        "translation-invariant",
        "score",
    ]
    // TODO: not scale-invariant??
    "procrustes-euclidean": properties: [
        "permutation-invariant",
        "rotation-invariant",
        "metric",
    ]
    "procrustes-angular": properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scale-invariant",
        "metric",
    ]
    "shape_metric-angular": properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scaling-invariant",
        "metric"
    ]
    rsa: properties: [
        "permutation-invariant",
        "scale-invariant",
        "translation-invariant",
        "score"
    ]
    cka: properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scale-invariant",
        "translation-invariant",
        "score"
    ]
    gs: properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scale-invariant",
        "translation-invariant",
    ]
    imd: properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scale-invariant",
        "translation-invariant",
    ]
    rtd: properties: [
        "permutation-invariant",
        "rotation-invariant",
        "scale-invariant",
        "translation-invariant",
    ]
}
