from functools import partial
from pathlib import Path
import re
from scipy.spatial.distance import cdist
import numpy as np

import similarity


def standardize_names(measures):
    thingsvision_mapping = {
        "cka_kernel_linear_unbiased": "cka-kernel=linear-hsic=song-score",
        "cka_kernel_linear_biased": "cka-kernel=linear-hsic=gretton-score",
        "cka_kernel_rbf_unbiased_sigma_1.0": "cka-kernel=(rbf-sigma={sigma})-hsic=song-score",
        "cka_kernel_rbf_biased_sigma_1.0": "cka-kernel=(rbf-sigma={sigma})-hsic=gretton-score",
    }
    for k in measures.keys():
        if "thingsvision/rsa" not in k or len(k.split("/")) != 2:
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

        "pwcca": "pwcca-score",
        "cca": "cca-score",
        "cca_squared_correlation": "cca-squared_score",
        "svcca": "svcca-score",
    }

    correct_cka_alignment_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "cka_debiased": "cka-kernel=linear-hsic=song-score",
    }

    repsim_mapping = {
        "AngularCKA.linear": "cka-kernel=linear-hsic=gretton-distance=angular",
        "AngularCKA.unb.linear": "cka-kernel=linear-hsic=lange-distance=angular",
        "AngularCKA.SqExp[{sigma}]": "cka-kernel=(rbf-sigma={sigma})-hsic=gretton-distance=angular",
        "AngularCKA.unb.SqExp[{sigma}]": "cka-kernel=(rbf-sigma={sigma})-hsic=lange-distance=angular",
        "AngularCKA.Laplace[{sigma}]": "cka-kernel=(laplace-sigma={sigma})-hsic=gretton-distance=angular",
        "AngularCKA.unb.Laplace[{sigma}]": "cka-kernel=(laplace-sigma={sigma})-hsic=lange-distance=angular",
        "ShapeMetric[{alpha}][angular]": "shape_metric-alpha={alpha}-distance=angular",
        "ShapeMetric[{alpha}][euclidean]": "shape_metric-alpha={alpha}-distance=euclidean",
    }

    nn_similarity_index_mapping = {
        "euclidean": "euclidean",
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "nbs": "nbs-score",
        "bures_distance": "bures_distance",
    }

    fsd_mapping = {
        "linear_CKA_loss": "cka-kernel=linear-hsic=gretton-score-negative_log",
        "linear_CKA": "cka-kernel=linear-hsic=gretton-score",
    }

    imd_mapping = {
        "imd": "imd",
    }

    ensd_mapping = {
        "ensd": "ensd",
        "computeDist": "ensd-distance=angular_normalized",
    }

    # TODO: parameters
    platonic_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "unbiased_cka": "cka-kernel=linear-hsic=song-score",
        "cka_rbf": "cka-kernel=(rbf-sigma={sigma})-hsic=gretton-score",
        "unbiased_cka_rbf": "cka-kernel=(rbf-sigma={sigma})-hsic=song-score",
        "cycle_knn_topk": "cycle_knn-topk={topk}",
        "mutual_knn_topk": "mutual_knn-topk={topk}",
        "lcs_knn_topk": "lcs_knn-topk={topk}",
        "cknna_topk": "cknna-topk={topk}",
        "svcca": "svcca-dim={dim}-score",
        "edit_distance_knn_topk": "edit_distance_knn-topk={topk}",
    }

    representation_similarity_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "cka_debiased": "cka-kernel=linear-hsic=song-score",
        "cca": "cca-squared_score",
    }

    resi_mapping = {
        "GeometryScore": "geometry_score",
        "PWCCA": "pwcca-score",
        "SVCCA": "svcca-var=0.99-score",
        "HardCorrelationMatch": "hard_correlation_match",
        "SoftCorrelationMatch": "soft_correlation_match",
        "DistanceCorrelation": "distance_correlation",
        "EigenspaceOverlapScore": "eigenspace_overlap_score",
        "IMDScore": "imd",
        "Gulp": "gulp",
        "LinearRegression": "linear_regression",
        "JaccardSimilarity": "jaccard_similarity",
        "RankSimilarity": "rank_similarity",
        "SecondOrderCosineSimilarity": "second_order_cosine_similarity",
        "AlignedCosineSimilarity": "aligned_cosine_similarity",
        "OrthogonalAngularShapeMetricCentered": "orthogonal_angular_shape_metric_centered",
        "OrthogonalProcrustesCenteredAndNormalized": "orthogonal_procrustes_centered_and_normalized",
        "PermutationProcrustes": "permutation_procrustes",
        "ProcrustesSizeAndShapeDistance": "procrustes-distance=euclidean",
        "RSMNormDifference": "rsm_norm_difference",
        "ConcentricityDifference": "concentricity_difference",
        "MagnitudeDifference": "magnitude_difference",
        "UniformityDifference": "uniformity_difference",

        "CKA": "cka-kernel=linear-hsic=gretton-score",

        # rsa
        "RSA_correlation_spearman": "rsa-rdm=correlation-compare=spearman",
        "RSA_correlation_euclidean": "rsa-rdm=correlation-compare=euclidean",
        "RSA_euclidean_spearman": "rsa-rdm=euclidean-compare=spearman",
        "RSA_euclidean_euclidean": "rsa-rdm=euclidean-compare=euclidean",
    }

    sim_metric_mapping = {
        "mean_cca_corr": "cca-score",
        "mean_sq_cca_corr": "cca-squared_score",
        "pwcca_dist": "pwcca-distance=one_minus_score",
        "lin_cka_dist": "cka-kernel=linear-hsic=gretton-distance=one_minus_score",
        "procrustes": "procrustes-distance=squared_euclidean",
    }

    svcca_mapping = {
        "cca": "cca-score",
        "cca_squared_correlation": "cca-squared_score",
        "pwcca": "pwcca-score",
        "pls": "pls",
    }

    drfrankenstein_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "cca": "cca-squared_score",
        "cka_debiased": "cka-kernel=linear-hsic=song-score",
    }

    implicitdeclaration_similarity_mapping = {
        "linear_cka": "cka-kernel=linear-hsic=gretton-score",
    }

    nnsrm_neurips18_mapping = {
        "rsa": "rsa-rdm=correlation-compare=pearson",
        "procrustes": "procrustes-distance=euclidean",  # TODO: what is this measure?
        "isc": "isc",
    }

    survey_measures_mapping = {
        "procrustes": "procrustes-distance=squared_euclidean",
    }

    llm_repsim_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "jaccard_similarity": "jaccard_similarity",
        "top_k_neighbors": "top_k_neighbors",
        "orthogonal_procrustes": "procrustes-distance=euclidean",
        "aligned_cossim": "aligned_cossim",
        "representational_similarity_analysis": "rsa-rdm=correlation-compare=pearson",
        "rsm_norm_diff": "rsm_norm_difference",
    }

    rtd_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
        "pwcca": "pwcca-score",
        "svcca": "svcca-score",
    }

    brain_language_nlp_mapping = {
        "ridge": "ridge-lambda=1-r2",
        "kernel_ridge": "kernel_ridge-lambda=1-r2",
    }

    brainscore_mapping = {
        "rsa-correlation-spearman": "rsa-rdm=correlation-compare=spearman",
        "correlation": "correlation",
        "cka": "cka-kernel=linear-hsic=gretton-score",

        "linear_regression-pearsonr_correlation": "linear_regression-pearsonr",
        "linear_regression-pearsonr_correlation-5folds_cv": "linear_regression-pearsonr-cv=5folds",

        "ridge_regression-pearsonr_correlation": "ridge-lambda=1-pearsonr",
        "ridge_regression-pearsonr_correlation-5folds_cv": "ridge-lambda=1-pearsonr-cv=5folds",

        "pls_regression-pearsonr_correlation": "pls-pearsonr",
        "pls_regression-pearsonr_correlation-5folds_cv": "pls-pearsonr-cv=5folds",
    }

    deepdive_mapping = {
        "neural_regression-alpha1-pearson_r-5folds_cv": "ridge-lambda=1-pearsonr-cv=5folds",
        "neural_regression-alpha1-pearson_r2-5folds_cv": "ridge-lambda=1-pearsonr2-cv=5folds",
        "neural_regression-alpha1-r2-5folds_cv": "ridge-lambda=1-r2-cv=5folds",

        "neural_regression-alpha0-pearson_r-5folds_cv": "linear_regression-pearsonr-cv=5folds",
        "neural_regression-alpha0-pearson_r2-5folds_cv": "linear_regression-pearsonr2-cv=5folds",
        "neural_regression-alpha0-r2-5folds_cv": "linear_regression-r2-cv=5folds",
    }

    neuroaimetrics_mapping = {
        "CKA": "cka-kernel=linear-hsic=gretton-score",
        "RSA": "rsa-rdm=correlation-compare=kendall",
        "SoftMatching": "soft_correlation_match",
        "LinearShapeMetric": "shape_metric-alpha={alpha}-distance=angular-cv=5folds",
        "VERSA": "versa",
        "pairwisematching": "pairwisematching",

        # TODO: ridge reg: alphas=np.logspace(-8,8,17)
        # parameters are hardcoded in the orginal (can't overwrite them)
        "LinearPredictivity": "ridge-pearsonr-cv=5folds",
        "reverseLinearPredictivity": "ridge-pearsonr-cv=5folds-reverse",
        "PLSreg": "pls-pearsonr-components=25-cv=5folds",
    }

    mouse_vision_mapping = {
        "rsa": "rsa-rdm=correlation-compare=pearson",
        "PLSNeuralMap": "pls-pearsonr-components={n_components}",
        "CorrelationNeuralMap": "correlation",
    }

    modelsym_mapping = {
        "wreath_cka": "wreath_cka",
        "wreath_procrustes": "wreath_procrustes",
        "ortho_cka": "cka-kernel=linear-hsic=gretton-score",
        "ortho_procrustes": "procrustes-distance=euclidean",
    }

    pyrcca_mapping = {
        "cca": "cca-score",
    }

    unsupervised_analysis_mapping = {
        "cka": "cka-kernel=linear-hsic=gretton-score",
    }

    rsatoolbox_mapping = {}
    for k in measures.keys():
        # process keys only of the form "rsatoolbox/*"
        if "rsatoolbox" not in k or len(k.split("/")) != 2:
            continue
        k = k.split("rsatoolbox/")[1]
        rdm_method = k.split("-")[1]
        compare_method = k.split("-")[-1]

        if rdm_method == "euclidean":
            rdm_method = "squared_euclidean"

        if compare_method == "corr":
            compare_method = "pearson"

        # TODO: is rho-a, rho-b same as spearman?
        # from docs: 'rho-a' = spearman correlation without tie correction
        # https://github.com/rsagroup/rsatoolbox/blob/main/src/rsatoolbox/rdm/compare.py

        rsatoolbox_mapping[k] = f"rsa-rdm={rdm_method}-compare={compare_method}"

    diffscore_keys = [k.split("/")[1] for k in measures.keys() if "diffscore/" in k]
    diffscore_mapping = {k: k for k in diffscore_keys}


    mapping = {
        "thingsvision": thingsvision_mapping,
        "netrep": netrep_mapping,
        "contrasim": contrasim_mapping,
        "correcting_cka_alignment": correct_cka_alignment_mapping,
        "repsim": repsim_mapping,
        "nn_similarity_index": nn_similarity_index_mapping,
        "rsatoolbox": rsatoolbox_mapping,
        "fsd": fsd_mapping,
        "ensd": ensd_mapping,
        "platonic": platonic_mapping,
        "representation_similarity": representation_similarity_mapping,
        "resi": resi_mapping,
        "sim_metric": sim_metric_mapping,
        "svcca": svcca_mapping,
        "drfrankenstein": drfrankenstein_mapping,
        "implicitdeclaration_similarity": implicitdeclaration_similarity_mapping,
        "nnsrm_neurips18": nnsrm_neurips18_mapping,
        "survey_measures": survey_measures_mapping,
        "llm_repsim": llm_repsim_mapping,
        "rtd": rtd_mapping,
        "brain_language_nlp": brain_language_nlp_mapping,
        "brainscore": brainscore_mapping,
        "deepdive": deepdive_mapping,
        "neuroaimetrics": neuroaimetrics_mapping,
        "mouse_vision": mouse_vision_mapping,
        "modelsym": modelsym_mapping,
        "pyrcca": pyrcca_mapping,
        "unsupervised_analysis": unsupervised_analysis_mapping,
        "imd": imd_mapping,
        # "diffscore": diffscore_mapping,  # TODO: too many measures
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
    # (Generalized Shape Metrics on Neural Representations. Williams et al., 2021)
    # take the arccosine to get angular distance
    {"inp": lambda k: k.endswith("-score"), "out": lambda k, v: (k.replace("-score", "-distance=angular"), v), "postprocessing": ["arccos"]},
    {"inp": lambda k: k.endswith("-distance=angular"), "out": lambda k, v: (k.replace("-distance=angular", "-score"), v), "postprocessing": ["cos"]},
    # shape metric with alpha=0 <=> cca
    {
        "inp": lambda k: "shape_metric-alpha={alpha}" in k,
        # "out": lambda k, v: (k.replace("shape_metric-alpha={alpha}", "cca"), wrap_shape_metric(v, alpha=0))
        "out": lambda k, v: (k.replace("shape_metric-alpha={alpha}", "cca"), partial(v, alpha=0))
    },
    # shape metric with alpha=1 <=> procrustes
    {
        "inp": lambda k: "shape_metric-alpha={alpha}" in k,
        "out": lambda k, v: (k.replace("shape_metric-alpha={alpha}", "procrustes"), partial(v, alpha=1))
    },
    # convert between euclidean and angular distance
    {
        "inp": lambda k: k.endswith("distance=euclidean"),
        "out": lambda k, v: (k.replace("distance=euclidean", "distance=angular"), v),
        "postprocessing": [
            {"id": "euclidean_to_angular_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    },
    {
        "inp": lambda k: k.endswith("distance=angular"),
        "out": lambda k, v: (k.replace("distance=angular", "distance=euclidean"), v),
        "postprocessing": [
            {"id": "angular_to_euclidean_shape_metric", "inputs": ["X", "Y", "score"]},
        ]
    },
    # squared euclidean <=> euclidean
    {
        "inp": lambda k: k.endswith("distance=squared_euclidean"),
        "out": lambda k, v: (k.replace("distance=squared_euclidean", "distance=euclidean"), v),
        "postprocessing": ["sqrt"]
    },
    {
        "inp": lambda k: k.endswith("distance=euclidean"),
        "out": lambda k, v: (k.replace("distance=euclidean", "distance=squared_euclidean"), v),
        "postprocessing": ["square"]
    },

    # Duality of Bures and Shape Distances with Implications for Comparing Neural Representations (Harvey et al., 2023)
    # Procrustes angular distance = arccos(NBS)
    {
        "inp": lambda k: "/nbs-distance=angular" in k,
        "out": lambda k, v: (k.replace("/nbs-distance=angular", "/procrustes-distance=angular"), v),
    },
    {
        "inp": lambda k: "/procrustes-distance=angular" in k,
        "out": lambda k, v: (k.replace("/procrustes-distance=angular", "/nbs-distance=angular"), v),
    },

    # (Ding, 2021) defines CKA "distance" as 1 - CKA
    {
        "inp": lambda k: k.endswith("distance=one_minus_score"),
        "out": lambda k, v: (
            k.replace("distance=one_minus_score", "score"),
            lambda X, Y, **kwargs: 1 - v(X, Y, **kwargs)
        )
    },

    # SVCCA = PCA + CCA
    {"inp": lambda k: "/cca" in k, "out": lambda k, v: (k.replace("/cca", "/svcca-dim=10"), v), "preprocessing": ["pca-dim10"]},
    {"inp": lambda k: "/cca" in k, "out": lambda k, v: (k.replace("/cca", "/svcca-var=0.95"), v), "preprocessing": ["pca-var95"]},
    {"inp": lambda k: "/cca" in k, "out": lambda k, v: (k.replace("/cca", "/svcca-var=0.99"), v), "preprocessing": ["pca-var99"]},

    # 
    # https://rsatoolbox.readthedocs.io/en/stable/comparing.html#whitened-comparison-measures
    # rsa-rdm=squared_euclidean-compare=cosine_cov <=> cka-kernel=linear-hsic=gretton-score
    {"inp": lambda k: "rsa-rdm=squared_euclidean-compare=cosine_cov" in k, "out": lambda k, v: (k.replace("rsa-rdm=squared_euclidean-compare=cosine_cov", "cka-kernel=linear-hsic=gretton-score"), v)},

]

# rdm
transforms.extend([
    # rdm/*/euclidean => rdm/*/squared_euclidean
    {
        "inp": lambda k: bool(re.match(r"^rdm/[^/]+/euclidean$", k)),
        "out": lambda k, v: (
            k.replace("/euclidean", "/squared_euclidean"),
            lambda X: v(X)**2
        )
    },
    # rdm/*/squared_euclidean => rdm/*/euclidean
    {
        "inp": lambda k: bool(re.match(r"^rdm/[^/]+/squared_euclidean$", k)),
        "out": lambda k, v: (
            k.replace("/squared_euclidean", "/euclidean"),
            lambda X: np.sqrt(v(X))
        )
    },
    # rdm/*/*_normalized => rdm/*/*
    {
        "inp": lambda k: bool(re.match(r"^rdm/[^/]+/[^/]+_normalized$", k)),
        "out": lambda k, v: (
            k.replace("_normalized", ""),
            lambda X: v(X) * X.shape[1]
        )
    },
    # rdm/*/* (without '_normalized') => rdm/*/*_normalized
    {
        "inp": lambda k: bool(re.match(r"^rdm/[^/]+/[^/]+$", k)) and "normalized" not in k,
        "out": lambda k, v: (
            k + "_normalized",
            lambda X: v(X) / X.shape[1]
        )
    }
])



# distance
transforms.extend([
    # distance/*/angular => measure/*/cosine
    {
        "inp": lambda k: bool(re.match(r"^distance/[^/]+/angular$", k)),
        "out": lambda k, v: (
            k.replace("distance/", "measure/").replace("/angular", "/cosine"),
            lambda X, Y, **kwargs: np.cos(v(X, Y, **kwargs))
        )
    },
    # measure/*/cosine => distance/*/angular
    {
        "inp": lambda k: bool(re.match(r"^measure/[^/]+/cosine$", k)),
        "out": lambda k, v: (
            k.replace("measure/", "distance/").replace("/cosine", "/angular"),
            lambda X, Y, **kwargs: np.arccos(v(X, Y, **kwargs))
        )
    }
])


# kernel
transforms.extend([
    # kernel/*/* (without -centered) => kernel/*/*-centered
    {
        "inp": lambda k: bool(re.match(r"^kernel/[^/]+/[^/]+$", k)) and "-centered" not in k,
        "out": lambda k, v: (
            k.replace("kernel/", "kernel/") + "-centered",
            lambda X: similarity.make("preprocessing/center_rows_columns")(v(X))
        )
    },
    # kernel/*/* (without -zero_diagonal) => kernel/*/*-zero_diagonal
    {
        "inp": lambda k: bool(re.match(r"^kernel/[^/]+/[^/]+$", k)) and "-zero_diagonal" not in k,
        "out": lambda k, v: (
            k.replace("kernel/", "kernel/") + "-zero_diagonal",
            lambda X: v(X) - np.diag(np.diag(v(X)))
        )
    },    
])

# derive rbf kernel from linear kernel
# TODO: keep this?
def sqrt_rbf_kernel(X, sigma):
    K = np.exp(-cdist(X, X)**2 / (2 * sigma**2))
    # centering
    H = np.eye(K.shape[0]) - 1/K.shape[0]
    K = H @ K @ H
    # take the matrix square root
    Uk, Sk, VkT = np.linalg.svd(K)
    K_sqrt = Uk @ np.diag(np.sqrt(np.clip(Sk, a_min=0, a_max=np.inf))) @ VkT
    K_recon = K_sqrt @ K_sqrt.T
    assert np.allclose(K, K_recon, atol=1e-10), np.max(np.abs(K - K_recon))
    return K_sqrt

transforms.extend([
    # measure/*/*-kernel=linear-* => measure/*/*-kernel=(rbf-sigma={sigma})-*
    {
        "inp": lambda k: "measure/" in k and "kernel=linear" in k,
        "out": lambda k, v: (
            k.replace("kernel=linear", "kernel=(rbf-sigma={sigma})"),
            lambda X, Y, sigma=1.0, **kwargs: v(sqrt_rbf_kernel(X, sigma), sqrt_rbf_kernel(Y, sigma), **kwargs)
        )
    }
])



# convert between kernel and rdm (see https://openreview.net/pdf?id=zMdnnFasgC)
def kernel_to_rdm(X, K):
    # Dij = Kii + Kjj - 2Kij
    return np.diag(K)[:, None] + np.diag(K)[None, :] - 2 * K


def rdm_to_kernel(X, D, q=2):
    # assume D_ij = ||x_i - x_j||^q_2
    # K_ij = 1/2 * (||x_i||^q_2 + ||x_j||^q_2 - ||x_i - x_j||^q_2)
    K = 1/2 * (np.sum(X**q, axis=1)[:, None] + np.sum(X**q, axis=1)[None, :] - D)
    return K


transforms.extend([
    # kernel/*/linear => rdm/*/squared_euclidean
    {
        "inp": lambda k: bool(re.match(r"^kernel/[^/]+/linear$", k)),
        "out": lambda k, v: (
            k.replace("kernel/", "rdm/").replace("/linear", "/squared_euclidean"),
            lambda X: kernel_to_rdm(X, v(X))
        )
    },
    # rdm/*/squared_euclidean => kernel/*/linear
    {
        "inp": lambda k: bool(re.match(r"^rdm/[^/]+/squared_euclidean$", k)),
        "out": lambda k, v: (
            k.replace("rdm/", "kernel/").replace("/squared_euclidean", "/linear"),
            lambda X: rdm_to_kernel(X, v(X), q=2)
        )
    }
])


# "fsd" repository defines CKA loss as -log(CKA)
transforms.extend([
    # measure/*/*-score-negative_log => measure/*/*-score
    {
        "inp": lambda k: bool(re.match(r"^measure/[^/]+/[^/]+-score-negative_log$", k)),
        "out": lambda k, v: (
            k.replace("-score-negative_log", "-score"),
            lambda X, Y, **kwargs: np.exp(-v(X, Y, **kwargs))
        )
    }
])

# TODO: results don't match exactly
# derive CKA from ENSD (https://www.biorxiv.org/content/10.1101/2023.07.27.550815v1.full.pdf)
transforms.extend([
    # measure/*/ensd-distance=angular_normalized => measure/*/cka-kernel=linear-hsic=gretton-score
    {
        "inp": lambda k: bool(re.match(r"^measure/[^/]+/ensd-distance=angular_normalized$", k)),
        "out": lambda k, v: (
            k.replace("ensd-distance=angular_normalized", "cka-kernel=linear-hsic=gretton-score"),
            lambda X, Y, **kwargs: np.cos(np.pi / 2 * v(X, Y, **kwargs))
        )
    }
])


compositions = []

def derive_cka(registry: dict):
    """
    Derive the Lange version of CKA from the gretton version.
    The Lange version removed the diagonal of the kernels before computing the cosine similarity.
    """
    derived_measures = {}
    repos = {key.split('/')[1] for key in registry if key.startswith('kernel/')}

    possible_requirements = [
        # the centered and zero_diagonal can either be in the kernel or the measure
        {"kernel": "linear-centered-zero_diagonal", "measure": "cosine"},
        {"kernel": "linear-centered", "measure": "zero_diagonal-cosine"},
        {"kernel": "linear", "measure": "centered-zero_diagonal-cosine"},
    ]

    for repo in repos:
        for requirement in possible_requirements:
            kernel_id = f"kernel/{repo}/{requirement['kernel']}"
            measure_id = f"measure/{repo}/{requirement['measure']}"

            if not (kernel_id in registry and measure_id in registry):
                continue

            # make sure the derived measure doesn't already exist
            new_measure_id = f"measure/{repo}/cka-kernel=linear-hsic=lange-score"
            if new_measure_id in registry:
                continue
                
            def _measure(X, Y, k=kernel_id, m=measure_id):
                Kx = registry[k](X)
                Ky = registry[k](Y)
                return registry[m](Kx, Ky)
                
            derived_measures[new_measure_id] = _measure
        
    return derived_measures


def derive_rsa_from_cka(registry: dict):
    derived_measures = {}
    repos = {key.split('/')[1] for key in registry if key.startswith('kernel/')}

    possible_requirements = [
        {"rdm": "squared_euclidean", "measure": "cosine"},
    ]
    for repo in repos:
        for requirement in possible_requirements:
            rdm_id = f"rdm/{repo}/{requirement['rdm']}"
            measure_id = f"measure/{repo}/{requirement['measure']}"

            if not (rdm_id in registry and measure_id in registry):
                continue

            new_measure_id = f"measure/{repo}/rsa-rdm={requirement['rdm']}-compare={requirement['measure']}"
            if new_measure_id in registry:
                continue

            def _measure(X, Y, r=rdm_id, m=measure_id):
                rdmX = registry[r](X)
                rdmY = registry[r](Y)
                return registry[m](rdmX, rdmY)
            
            derived_measures[new_measure_id] = _measure
        
    return derived_measures


compositions.append(derive_cka)
compositions.append(derive_rsa_from_cka)


def derive_measures(measures, transforms, compositions=None):
    """
    Automatically derive measures from a set of measures and a set of transforms.
    The transforms apply to a single measure at a time. If you want to derive measures
    by composing multiple existing measures, you can do so with the compositions argument.

    A transform is more specific. It has an input condition on the measure id and an output function that takes
    the measure id and the measure itself as input and returns a new measure id and the new measure.
    A composition is more general. It takes the entire dictionary of measures as input and return a dictionary
    of new measures.

    Args:
        measures (dict): a dictionary of measures.
        transforms (list): a list of transforms.
        compositions (list): a list of compositions.

    Returns:
        dict: a dictionary of derived measures.
    """

    def _derive_once(measures, transforms, compositions):
        derived_measures = {}

        # apply transforms
        for transform in transforms:
            for measure_id, measure in measures.items():
                if not transform["inp"](measure_id):
                    continue

                new_measure_id, new_measure = transform["out"](measure_id, measure)

                if new_measure_id in measures:
                    continue

                if transform.get("postprocessing", None) or transform.get("preprocessing", None):
                    new_measure = similarity.wrap_measure(
                        new_measure,
                        preprocessing=transform.get("preprocessing", None),
                        postprocessing=transform.get("postprocessing", None)
                    )

                print(f"Derived measure: {new_measure_id}")
                derived_measures[new_measure_id] = new_measure

        # apply compositions
        if compositions is not None:
            for composition in compositions:
                new_measures = composition(measures)
                print("Add compositions:", new_measures.keys())
                derived_measures.update(new_measures)

        return derived_measures

    # derive measures until no new measures are derived
    derived = {}
    while True:
        new_derived = _derive_once(measures, transforms, compositions)
        if len(new_derived) == 0:
            break
        derived = {**derived, **new_derived}  # only the derived measures
        measures = {**measures, **new_derived}  # input measures + derived measures
    
    return derived


def register_standardized_measures():
    registry = similarity.registration.registry

    # TODO: remove papers
    registry = {k: v for k, v in registry.items() if "paper/" not in k}

    measures = standardize_names(registry)
    # add "measure/" prefix to the standardized measures
    measures = {f"measure/{k}": v for k, v in measures.items()}
    updated_registry = {**registry, **measures}  # don't modify directly the registry

    derived_measures = derive_measures(updated_registry, transforms, compositions)
    measures.update(derived_measures)

    for k, v in measures.items():
        similarity.register(k, v)

    # keep track of derived measures
    for k, v in derived_measures.items():
        similarity.registration.DERIVED_MEASURES[k] = v


if __name__ == "__main__":
    from similarity.plotting import plot_scores, plot_measures

    np.random.seed(0)

    save_dir = Path(__file__).parent.parent / "figures" / Path(__file__).stem

    # repos_to_plot = [
    #     "netrep",
    #     "rsatoolbox",
    #     "repsim",
    #     "contrasim",
    #     "correcting_cka_alignment",
    #     "thingsvision",
    #     "nn_similarity_index",
    #     "fsd",
    #     "ensd",
    #     "platonic",
    #     "representation_similarity",
    #     "resi",
    #     "sim_metric",
    #     "svcca",
    #     "drfrankenstein",
    #     "implicitdeclaration_similarity",
    #     "nnsrm_neurips18",
    #     "survey_measures",
    #     "llm_repsim",
    #     "rtd",
    #     "brain_language_nlp",
    # ]
    repos_to_plot = [
        "rsatoolbox",
        "representation_similarity",
        "nn_similarity_index",
        "netrep",
        "sim_metric",
        "repsim",
        "platonic",
        "neuroaimetrics",
        "resi"
        # "diffscore"
    ]
    repos_to_plot = None


    measures = similarity.all_measures()

    if repos_to_plot is not None:
        measures = {k: v for k, v in measures.items() if any(repo in k for repo in repos_to_plot)}
    
    # measures = {k: v for k, v in measures.items() if "nbs" in k or "procrustes" in k}
    # measures = {k: v for k, v in measures.items() if "cka" in k}
    # measures = {k: v for k, v in measures.items() if "cca" in k}
    # measures = {k: v for k, v in measures.items() if "rsa" in k}
    # measures = {k: v for k, v in measures.items() if "ridge" in k}

    # original = measures - derived
    original_measures = {k: v for k, v in measures.items() if k not in similarity.registration.DERIVED_MEASURES}
    derived_measures = similarity.registration.DERIVED_MEASURES

    if repos_to_plot is not None:
        original_measures = {k: v for k, v in original_measures.items() if any(repo in k for repo in repos_to_plot)}
        derived_measures = {k: v for k, v in derived_measures.items() if any(repo in k for repo in repos_to_plot)}


    plot_measures(original_measures, derived_measures=derived_measures, save_dir=save_dir)



    # for all measures with parameter 'sigma={sigma}', create a new measure with 'sigma=1.0'
    for k, v in list(measures.items()):
        if 'sigma={sigma}' in k:
            new_k = k.replace('sigma={sigma}', 'sigma=1.0')
            measures[new_k] = partial(v, sigma=1.0)

    # filter measures that don't have parameters
    measures = {k: v for k, v in measures.items() if '{' not in k}

    plot_scores(measures, save_dir=save_dir)
