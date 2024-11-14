from pathlib import Path
import similarity


def standardize_names(measures):
    thingsvision_mapping = {
        "cka_kernel_linear_unbiased": "cka-kernel=linear-hsic=song-score",
        "cka_kernel_linear_biased": "cka-kernel=linear-hsic=gretton-score",
        "cka_kernel_rbf_unbiased_sigma_1.0": "cka-kernel=(rbf-sigma=1.0)-hsic=song-score",
        "cka_kernel_rbf_biased_sigma_1.0": "cka-kernel=(rbf-sigma=1.0)-hsic=gretton-score",
    }
    for k in measures.keys():
        if "thingsvision/rsa" not in k:
            continue
        k = k.split("thingsvision/")[1]
        # extract rsa and corr method
        rsa_method = k.split("rsa_method_")[1].split("_corr_method_")[0]
        corr_method = k.split("corr_method_")[1]
        thingsvision_mapping[k] = f"rsa-rdm={rsa_method}-compare={corr_method}"

    netrep_mapping = {
        "LinearCKA": "cka-kernel=linear-hsic=gretton-distance=angular",
        "LinearMetric_angular": "shape_metric-alpha={alpha}-distance=angular",
        "LinearMetric_euclidean": "shape_metric-alpha={alpha}-distance=euclidean",
        "PermutationMetric_angular": "permutation_metric-distance=angular",
        "PermutationMetric_euclidean": "permutation_metric-distance=euclidean",
    }

    contrasim_mapping = {
        "feature_space_linear_cka": "cka-kernel=linear-hsic=gretton-score",
        "feature_space_linear_cka_debiased": "cka-kernel=linear-hsic=song-score",
        "pwcca": "pwcca",
        "cca": "cca-score",
        "svcca": "svcca",
    }

    correct_cka_alignment_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "cka_debiased": "cka-kernel=linear-hsic=song-score",
    }

    repsim_mapping = {
        "AngularCKA.linear": "cka-kernel=linear-hsic=gretton-distance=angular",
        "AngularCKA.unb.linear": "cka-kernel=linear-hsic=lange-distance=angular",
        "AngularCKA.SqExp[{sigma}]": "cka-kernel=rbf-sigma={sigma}-hsic=lange-distance=angular",
        "AngularCKA.unb.SqExp[{sigma}]": "cka-kernel=rbf-sigma={sigma}-hsic=lange-distance=angular",
        "AngularCKA.Laplace[{sigma}]": "cka-kernel=laplace-hsic=lange-distance=angular",
        "AngularCKA.unb.Laplace[{sigma}]": "cka-kernel=laplace-hsic=lange-distance=angular",
        "ShapeMetric[{alpha}][angular]": "shape_metric-alpha={alpha}-distance=angular",
        "ShapeMetric[{alpha}][euclidean]": "shape_metric-alpha={alpha}-distance=euclidean",
    }

    rsatoolbox_mapping = {}
    for k in measures.keys():
        if "rsatoolbox" not in k:
            continue
        k = k.split("rsatoolbox/")[1]
        rdm_method = k.split("-")[1]
        compare_method = k.split("-")[-1]
        rsatoolbox_mapping[k] = f"rsa-rdm={rdm_method}-compare={compare_method}"

    mapping = {
        "thingsvision": thingsvision_mapping,
        "netrep": netrep_mapping,
        "contrasim": contrasim_mapping,
        "correcting_cka_alignment": correct_cka_alignment_mapping,
        "repsim": repsim_mapping,
        "rsatoolbox": rsatoolbox_mapping,
    }

    standardized_measures = {}
    for key, value in measures.items():
        if len(key.split("/")) != 2:
            continue

        measure_name = key.split("/")[-1]
        repo_name = key.split("/")[-2]

        if repo_name not in mapping:
            continue
        
        print(repo_name, measure_name)
        new_name = mapping[repo_name][measure_name]
        standardized_measures[f"{repo_name}/{new_name}"] = value

    return standardized_measures


transforms = [
    # Generalized Shape Metrics on Neural Representations (Williams et al., 2021)
    # take the arccosine to get angular distance
    {"inp": lambda k: "score" in k, "out": lambda k, v: (k.replace("score", "distance=angular"), v), "postprocessing": ["arccos"]},
    {"inp": lambda k: "distance=angular" in k, "out": lambda k, v: (k.replace("distance=angular", "score"), v), "postprocessing": ["cos"]},

    # convert between euclidean and angular distance
    {
        "inp": lambda k: "distance=euclidean" in k,
        "out": lambda k, v: (k.replace("distance=euclidean", "distance=angular"), v),
        "postprocessing": [
            {"id": "euclidean_to_angular_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    },
    {
        "inp": lambda k: "distance=angular" in k,
        "out": lambda k, v: (k.replace("distance=angular", "distance=euclidean"), v),
        "postprocessing": [
            {"id": "angular_to_euclidean_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    }
]

def wrap_shape_metric(measure, alpha):
    def _measure(X, Y):
        return measure(X, Y, alpha=alpha)
    return _measure

transforms.append({
    "inp": lambda k: "shape_metric-alpha={alpha}" in k,
    "out": lambda k, v: (k.replace("shape_metric-alpha={alpha}", "cca"), wrap_shape_metric(v, alpha=0))
})


def derive_measures(measures):
    def _derive_once(measures):
        derived_measures = {}
        for transform in transforms:
            for measure_id, measure in measures.items():
                if not transform["inp"](measure_id):
                    continue

                new_measure_id, measure = transform["out"](measure_id, measure)

                if new_measure_id in measures:
                    continue

                new_measure = similarity.wrap_measure(
                    measure,
                    preprocessing=transform.get("preprocessing", None),
                    postprocessing=transform.get("postprocessing", None)
                )
                print(f"Derived measure: {new_measure_id}")
                derived_measures[new_measure_id] = new_measure

        return derived_measures

    # derive measures until no new measures are derived
    derived = {}
    while True:
        new_derived = _derive_once(measures)
        if len(new_derived) == 0:
            break
        derived = {**derived, **new_derived}
        measures = {**measures, **new_derived}
    
    return derived


if __name__ == "__main__":
    measures = standardize_names(similarity.registration.registry)
    # add "measure/" prefix
    measures = {f"measure/{k}": v for k, v in measures.items()}

    derived_measures = derive_measures(measures)
    print(list(derived_measures.keys()))
    # breakpoint()
    measures.update(derived_measures)

    # filter measures that don't have parameters
    measures = {k: v for k, v in measures.items() if '{' not in k}

    from similarity.plotting import plot_scores

    save_dir = Path(__file__).parent.parent / "figures" / Path(__file__).stem
    plot_scores(measures, save_dir=save_dir)

