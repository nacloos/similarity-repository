from similarity import register, make

defaults = {
    "procrustes-angular-score":     "netrep",
    "nbs": "",
    "rsa-correlation-corr":         "rsatoolbox",
    # cka variants
    "cka-hsic_gretton":             "",
    "cka-hsic_gretton-angular":     "",
    "cka-hsic_song":                "",
    "cka-hsic_song-angular":        "",
    "cka-hsic_lange-angular":       "repsim",
    # cca variants
    "svcca-var99":                  "",
    "pwcca":                        "",
    "cca-mean_sq_corr":             "",
    "cca-angular-score":            "netrep",
    # linear regressino variants
    "pls-pearson_r#10splits_90/10ratio_cv": "",
    "pls-pearson_r#5folds_cv":      "",
}
