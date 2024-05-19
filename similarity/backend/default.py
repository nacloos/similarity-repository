from functools import partial
from similarity import register, make


defaults = {
    "procrustes-angular-score":         "netrep",
    "nbs":                              "netrep",
    "rsa-correlation-corr":             "rsatoolbox",
    # cka variants
    "cka-hsic_gretton":                 "kornblith19",
    "cka-hsic_gretton-angular-score":   "kornblith19",
    "cka-hsic_song":                    "kornblith19",
    "cka-hsic_song-angular":            "kornblith19",
    "cka-hsic_lange-angular":           "repsim",
    # cca variants
    "svcca-var99":                      "sim_metric",
    "pwcca":                            "sim_metric",
    "cca-mean_sq_corr":                 "sim_metric",
    "cca-angular-score":                "sim_metric",
    # linear regressino variants
    "pls-pearson_r#10splits_90/10ratio_cv": "brainscore",
    "pls-pearson_r#5folds_cv":          "brainscore",
}


# register all defaults
for measure, backend in defaults.items():
    register(
        f"measure.default.{measure}",
        partial(make, f"measure.{backend}.{measure}")
    )
