# TODO: deprecated
# from functools import partial
# from similarity import register, make, is_registered


# defaults = {
#     "procrustes-angular-score":         "netrep",
#     "nbs":                              "netrep",
#     "rsa-correlation-corr":             "rsatoolbox",
#     # cka variants
#     # "cka-hsic_gretton":                 "kornblith19",
#     # "cka-hsic_gretton-angular-score":   "kornblith19",
#     # "cka-hsic_song":                    "kornblith19",
#     # "cka-hsic_song-angular-score":      "kornblith19",
#     "cka-hsic_gretton":                 "yuanli2333",
#     "cka-hsic_gretton-angular-score":   "yuanli2333",
#     "cka-hsic_song":                    "kornblith19",
#     "cka-hsic_song-angular-score":      "kornblith19",
#     "cka-hsic_lange-angular-score":     "repsim",
#     # cca variants
#     "svcca-var99":                      "sim_metric",
#     "pwcca":                            "sim_metric",
#     "cca-mean_sq_corr":                 "sim_metric",
#     "cca-angular-score":                "sim_metric",
#     # linear regressino variants
#     "pls-pearson_r#10splits_90/10ratio_cv": "brainscore",
#     "pls-pearson_r#5folds_cv":          "brainscore",
# }


# TODO: error     "cka-hsic_song-angular-score":      "kornblith19",


# register all defaults
# for measure, backend in defaults.items():
#     try:
#         make(f"measure.{backend}.{measure}")
#     except:
#         print(f"Could not register {measure} from {backend}")
#         continue

#     register(
#         f"measure.default.{measure}",
#         partial(make, f"measure.{backend}.{measure}")
#     )
    