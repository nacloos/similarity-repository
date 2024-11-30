# Automatically generated code. Do not modify.
from typing import Literal


IdType = Literal[
	"contrasim/cca",
	"contrasim/cca_squared_correlation",
	"contrasim/feature_space_linear_cka",
	"contrasim/feature_space_linear_cka_debiased",
	"contrasim/pwcca",
	"contrasim/svcca",
	"correcting_cka_alignment/cka",
	"correcting_cka_alignment/cka_debiased",
	"distance/contrasim/angular",
	"distance/correcting_cka_alignment/angular",
	"distance/netrep/angular",
	"drfrankenstein/cca",
	"drfrankenstein/cka",
	"ensd/computeDist",
	"ensd/ensd",
	"fsd/linear_CKA_loss",
	"hsic/fsd/linear",
	"hsic/platonic/gretton",
	"hsic/platonic/song",
	"hsic/rtd/gretton",
	"implicitdeclaration_similarity/linear_cka",
	"kernel/contrasim/linear",
	"kernel/contrasim/linear-centered",
	"kernel/contrasim/linear-centered-zero_diagonal",
	"kernel/contrasim/linear-zero_diagonal",
	"kernel/contrasim/linear-zero_diagonal-centered",
	"kernel/correcting_cka_alignment/linear",
	"kernel/correcting_cka_alignment/linear-centered",
	"kernel/correcting_cka_alignment/linear-centered-zero_diagonal",
	"kernel/correcting_cka_alignment/linear-zero_diagonal",
	"kernel/correcting_cka_alignment/linear-zero_diagonal-centered",
	"kernel/netrep/linear",
	"kernel/netrep/linear-centered",
	"kernel/netrep/linear-centered-zero_diagonal",
	"kernel/netrep/linear-zero_diagonal",
	"kernel/netrep/linear-zero_diagonal-centered",
	"kernel/representation_similarity/linear",
	"kernel/representation_similarity/linear-centered",
	"kernel/representation_similarity/linear-centered-zero_diagonal",
	"kernel/representation_similarity/linear-zero_diagonal",
	"kernel/representation_similarity/linear-zero_diagonal-centered",
	"kernel/representation_similarity/rbf-threshold={threshold}",
	"kernel/rsatoolbox/linear",
	"kernel/rsatoolbox/linear-centered",
	"kernel/rsatoolbox/linear-centered-zero_diagonal",
	"kernel/rsatoolbox/linear-zero_diagonal",
	"kernel/rsatoolbox/linear-zero_diagonal-centered",
	"kernel/thingsvision/linear",
	"kernel/thingsvision/linear-centered",
	"kernel/thingsvision/linear-centered-zero_diagonal",
	"kernel/thingsvision/linear-zero_diagonal",
	"kernel/thingsvision/linear-zero_diagonal-centered",
	"kernel/thingsvision/rbf-sigma=1.0-centered",
	"kernel/thingsvision/rbf-sigma=1.0-centered-zero_diagonal",
	"kernel/thingsvision/rbf-sigma=1.0-zero_diagonal",
	"kernel/thingsvision/rbf-sigma=1.0-zero_diagonal-centered",
	"kernel/thingsvision/rbf-sigma={sigma}",
	"measure/brain_language_nlp/kernel_ridge",
	"measure/brain_language_nlp/kernel_ridge_svd",
	"measure/brain_language_nlp/ridge",
	"measure/brain_language_nlp/ridge_sk",
	"measure/brain_language_nlp/ridge_svd",
	"measure/brainscore",
	"measure/contrasim/cca-distance=angular",
	"measure/contrasim/cca-distance=euclidean",
	"measure/contrasim/cca-score",
	"measure/contrasim/cca-squared_score",
	"measure/contrasim/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/contrasim/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/contrasim/cka-kernel=linear-hsic=gretton-score",
	"measure/contrasim/cka-kernel=linear-hsic=lange-distance=angular",
	"measure/contrasim/cka-kernel=linear-hsic=lange-distance=euclidean",
	"measure/contrasim/cka-kernel=linear-hsic=lange-score",
	"measure/contrasim/cka-kernel=linear-hsic=song-distance=angular",
	"measure/contrasim/cka-kernel=linear-hsic=song-distance=euclidean",
	"measure/contrasim/cka-kernel=linear-hsic=song-score",
	"measure/contrasim/cosine",
	"measure/contrasim/pwcca-distance=angular",
	"measure/contrasim/pwcca-distance=euclidean",
	"measure/contrasim/pwcca-score",
	"measure/contrasim/rsa-rdm=squared_euclidean-compare=cosine",
	"measure/contrasim/svcca-distance=angular",
	"measure/contrasim/svcca-distance=euclidean",
	"measure/contrasim/svcca-score",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=gretton-score",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=lange-distance=angular",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=lange-distance=euclidean",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=lange-score",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=song-distance=angular",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=song-distance=euclidean",
	"measure/correcting_cka_alignment/cka-kernel=linear-hsic=song-score",
	"measure/correcting_cka_alignment/cosine",
	"measure/correcting_cka_alignment/rsa-rdm=squared_euclidean-compare=cosine",
	"measure/deepdive",
	"measure/deepdive/linreg-pearson_r-5folds_cv",
	"measure/deepdive/linreg-pearson_r2-5folds_cv",
	"measure/deepdive/linreg-r2-5folds_cv",
	"measure/deepdive/ridge-lambda1-pearson_r-5folds_cv",
	"measure/deepdive/ridge-lambda1-pearson_r2-5folds_cv",
	"measure/deepdive/ridge-lambda1-r2-5folds_cv",
	"measure/drfrankenstein/cca-distance=angular",
	"measure/drfrankenstein/cca-distance=euclidean",
	"measure/drfrankenstein/cca-score",
	"measure/drfrankenstein/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/drfrankenstein/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/drfrankenstein/cka-kernel=linear-hsic=gretton-score",
	"measure/ensd/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/ensd/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/ensd/cka-kernel=linear-hsic=gretton-score",
	"measure/ensd/ensd",
	"measure/ensd/ensd-distance=angular_normalized",
	"measure/fsd/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/fsd/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/fsd/cka-kernel=linear-hsic=gretton-score",
	"measure/fsd/cka-kernel=linear-hsic=gretton-score-negative_log",
	"measure/imd",
	"measure/imd/imd",
	"measure/implicitdeclaration_similarity/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/implicitdeclaration_similarity/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/implicitdeclaration_similarity/cka-kernel=linear-hsic=gretton-score",
	"measure/implicitdeclaration_similarity/cos_sparse_HSIC",
	"measure/implicitdeclaration_similarity/linear_sparse_HSIC",
	"measure/implicitdeclaration_similarity/rbf_sparse_HSIC",
	"measure/mklabunde",
	"measure/mklabunde/procrustes-sq-euclidean",
	"measure/netrep/cca-distance=angular",
	"measure/netrep/cca-distance=euclidean",
	"measure/netrep/cca-score",
	"measure/netrep/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/netrep/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/netrep/cka-kernel=linear-hsic=gretton-score",
	"measure/netrep/cka-kernel=linear-hsic=lange-distance=angular",
	"measure/netrep/cka-kernel=linear-hsic=lange-distance=euclidean",
	"measure/netrep/cka-kernel=linear-hsic=lange-score",
	"measure/netrep/cosine",
	"measure/netrep/nbs-distance=angular",
	"measure/netrep/nbs-distance=euclidean",
	"measure/netrep/nbs-score",
	"measure/netrep/permutation_metric-distance=angular",
	"measure/netrep/permutation_metric-distance=euclidean",
	"measure/netrep/permutation_metric-score",
	"measure/netrep/procrustes-distance=angular",
	"measure/netrep/procrustes-distance=euclidean",
	"measure/netrep/procrustes-score",
	"measure/netrep/rsa-rdm=squared_euclidean-compare=cosine",
	"measure/neuroaimetrics/CKA",
	"measure/neuroaimetrics/LinearPredictivity",
	"measure/neuroaimetrics/LinearShapeMetric",
	"measure/neuroaimetrics/PLSreg",
	"measure/neuroaimetrics/RSA",
	"measure/neuroaimetrics/SoftMatching",
	"measure/neuroaimetrics/VERSA",
	"measure/neuroaimetrics/pairwisematching",
	"measure/neuroaimetrics/reverseLinearPredictivity",
	"measure/nn_similarity_index/bures_distance",
	"measure/nn_similarity_index/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/nn_similarity_index/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/nn_similarity_index/cka-kernel=linear-hsic=gretton-score",
	"measure/nn_similarity_index/euclidean",
	"measure/nn_similarity_index/nbs-distance=angular",
	"measure/nn_similarity_index/nbs-distance=euclidean",
	"measure/nn_similarity_index/nbs-score",
	"measure/nn_similarity_index/procrustes-distance=angular",
	"measure/nn_similarity_index/procrustes-distance=euclidean",
	"measure/nn_similarity_index/procrustes-score",
	"measure/nnsrm_neurips18/isc",
	"measure/nnsrm_neurips18/nbs-distance=angular",
	"measure/nnsrm_neurips18/nbs-distance=euclidean",
	"measure/nnsrm_neurips18/nbs-score",
	"measure/nnsrm_neurips18/procrustes-distance=angular",
	"measure/nnsrm_neurips18/procrustes-distance=euclidean",
	"measure/nnsrm_neurips18/procrustes-score",
	"measure/nnsrm_neurips18/rsa-rdm=correlation-compare=spearman",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=angular",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=euclidean",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-score",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=song-distance=angular",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=song-distance=euclidean",
	"measure/platonic/cka-kernel=(rbf-sigma=1.0)-hsic=song-score",
	"measure/platonic/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/platonic/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/platonic/cka-kernel=linear-hsic=gretton-score",
	"measure/platonic/cka-kernel=linear-hsic=song-distance=angular",
	"measure/platonic/cka-kernel=linear-hsic=song-distance=euclidean",
	"measure/platonic/cka-kernel=linear-hsic=song-score",
	"measure/platonic/cknna-topk=10",
	"measure/platonic/cycle_knn-topk=10",
	"measure/platonic/edit_distance_knn-topk=10",
	"measure/platonic/lcs_knn-topk=10",
	"measure/platonic/mutual_knn-topk=10",
	"measure/platonic/svcca-distance=angular",
	"measure/platonic/svcca-distance=euclidean",
	"measure/platonic/svcca-score",
	"measure/pyrcca",
	"measure/representation_similarity/cca-squared_score",
	"measure/representation_similarity/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/representation_similarity/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/representation_similarity/cka-kernel=linear-hsic=gretton-score",
	"measure/representation_similarity/cka-kernel=linear-hsic=song-distance=angular",
	"measure/representation_similarity/cka-kernel=linear-hsic=song-distance=euclidean",
	"measure/representation_similarity/cka-kernel=linear-hsic=song-score",
	"measure/repsim/cca-distance=angular",
	"measure/repsim/cca-distance=euclidean",
	"measure/repsim/cca-score",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=angular",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=euclidean",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-score",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=lange-distance=angular",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=lange-distance=euclidean",
	"measure/repsim/cka-kernel=(rbf-sigma=1.0)-hsic=lange-score",
	"measure/repsim/cka-kernel=laplace-hsic=gretton-distance=angular",
	"measure/repsim/cka-kernel=laplace-hsic=gretton-distance=euclidean",
	"measure/repsim/cka-kernel=laplace-hsic=gretton-score",
	"measure/repsim/cka-kernel=laplace-hsic=lange-distance=angular",
	"measure/repsim/cka-kernel=laplace-hsic=lange-distance=euclidean",
	"measure/repsim/cka-kernel=laplace-hsic=lange-score",
	"measure/repsim/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/repsim/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/repsim/cka-kernel=linear-hsic=gretton-score",
	"measure/repsim/cka-kernel=linear-hsic=lange-distance=angular",
	"measure/repsim/cka-kernel=linear-hsic=lange-distance=euclidean",
	"measure/repsim/cka-kernel=linear-hsic=lange-score",
	"measure/repsim/nbs-distance=angular",
	"measure/repsim/nbs-distance=euclidean",
	"measure/repsim/nbs-score",
	"measure/repsim/procrustes-distance=angular",
	"measure/repsim/procrustes-distance=euclidean",
	"measure/repsim/procrustes-score",
	"measure/resi/aligned_cosine_similarity",
	"measure/resi/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/resi/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/resi/cka-kernel=linear-hsic=gretton-score",
	"measure/resi/concentricity_difference",
	"measure/resi/distance_correlation",
	"measure/resi/eigenspace_overlap_score",
	"measure/resi/gulp",
	"measure/resi/hard_correlation_match",
	"measure/resi/imd_score",
	"measure/resi/jaccard_similarity",
	"measure/resi/linear_regression",
	"measure/resi/magnitude_difference",
	"measure/resi/orthogonal_angular_shape_metric_centered",
	"measure/resi/orthogonal_procrustes_centered_and_normalized",
	"measure/resi/permutation_procrustes",
	"measure/resi/procrustes_size_and_shape_distance",
	"measure/resi/pwcca-distance=angular",
	"measure/resi/pwcca-distance=euclidean",
	"measure/resi/pwcca-score",
	"measure/resi/rank_similarity",
	"measure/resi/rsa-rdm=correlation-compare=euclidean",
	"measure/resi/rsa-rdm=correlation-compare=spearman",
	"measure/resi/rsa-rdm=euclidean-compare=euclidean",
	"measure/resi/rsa-rdm=euclidean-compare=spearman",
	"measure/resi/rsm_norm_difference",
	"measure/resi/second_order_cosine_similarity",
	"measure/resi/soft_correlation_match",
	"measure/resi/svcca-distance=angular",
	"measure/resi/svcca-distance=euclidean",
	"measure/resi/svcca-score",
	"measure/resi/uniformity_difference",
	"measure/rsatoolbox/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/rsatoolbox/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/rsatoolbox/cka-kernel=linear-hsic=gretton-score",
	"measure/rsatoolbox/cka-kernel=linear-hsic=lange-distance=angular",
	"measure/rsatoolbox/cka-kernel=linear-hsic=lange-distance=euclidean",
	"measure/rsatoolbox/cka-kernel=linear-hsic=lange-score",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=bures",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=bures_metric",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=corr",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=corr_cov",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=cosine",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=cosine_cov",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=kendall",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=rho_a",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=spearman",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=tau_a",
	"measure/rsatoolbox/rsa-rdm=correlation-compare=tau_b",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=bures",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=bures_metric",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=corr",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=corr_cov",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=cosine",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=cosine_cov",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=kendall",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=rho_a",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=spearman",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=tau_a",
	"measure/rsatoolbox/rsa-rdm=mahalanobis-compare=tau_b",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=bures",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=bures_metric",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=corr",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=corr_cov",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=cosine",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=cosine_cov",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=kendall",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=rho_a",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=spearman",
	"measure/rsatoolbox/rsa-rdm=poisson-compare=tau_b",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=bures",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=bures_metric",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=corr",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=corr_cov",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=cosine",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=kendall",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=rho_a",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=spearman",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=tau_a",
	"measure/rsatoolbox/rsa-rdm=squared_euclidean-compare=tau_b",
	"measure/rsatoolbox/zero_diagonal-cosine",
	"measure/rtd/cka",
	"measure/rtd/svcca",
	"measure/sim_metric/cca-distance=angular",
	"measure/sim_metric/cca-distance=euclidean",
	"measure/sim_metric/cca-score",
	"measure/sim_metric/cca-squared_score",
	"measure/sim_metric/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/sim_metric/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/sim_metric/cka-kernel=linear-hsic=gretton-distance=one_minus_score",
	"measure/sim_metric/cka-kernel=linear-hsic=gretton-score",
	"measure/sim_metric/nbs-distance=angular",
	"measure/sim_metric/nbs-distance=euclidean",
	"measure/sim_metric/nbs-score",
	"measure/sim_metric/procrustes-distance=angular",
	"measure/sim_metric/procrustes-distance=euclidean",
	"measure/sim_metric/procrustes-distance=squared_euclidean",
	"measure/sim_metric/procrustes-score",
	"measure/sim_metric/pwcca-distance=angular",
	"measure/sim_metric/pwcca-distance=euclidean",
	"measure/sim_metric/pwcca-distance=one_minus_score",
	"measure/sim_metric/pwcca-score",
	"measure/stir/kernel_CKA",
	"measure/stir/kernel_HSIC",
	"measure/stir/linear_CKA",
	"measure/stir/linear_HSIC",
	"measure/stir/unbiased_linear_HSIC",
	"measure/subspacematch",
	"measure/subspacematch/max_match",
	"measure/svcca/cca-distance=angular",
	"measure/svcca/cca-distance=euclidean",
	"measure/svcca/cca-score",
	"measure/svcca/cca-squared_score",
	"measure/svcca/pls",
	"measure/svcca/pwcca-distance=angular",
	"measure/svcca/pwcca-distance=euclidean",
	"measure/svcca/pwcca-score",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=angular",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-distance=euclidean",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=gretton-score",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=song-distance=angular",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=song-distance=euclidean",
	"measure/thingsvision/cka-kernel=(rbf-sigma=1.0)-hsic=song-score",
	"measure/thingsvision/cka-kernel=linear-hsic=gretton-distance=angular",
	"measure/thingsvision/cka-kernel=linear-hsic=gretton-distance=euclidean",
	"measure/thingsvision/cka-kernel=linear-hsic=gretton-score",
	"measure/thingsvision/cka-kernel=linear-hsic=song-distance=angular",
	"measure/thingsvision/cka-kernel=linear-hsic=song-distance=euclidean",
	"measure/thingsvision/cka-kernel=linear-hsic=song-score",
	"measure/thingsvision/rsa-rdm=correlation-compare=pearson",
	"measure/thingsvision/rsa-rdm=correlation-compare=spearman",
	"measure/thingsvision/rsa-rdm=cosine-compare=pearson",
	"measure/thingsvision/rsa-rdm=cosine-compare=spearman",
	"measure/thingsvision/rsa-rdm=euclidean-compare=pearson",
	"measure/thingsvision/rsa-rdm=euclidean-compare=spearman",
	"measure/thingsvision/rsa-rdm=gaussian-compare=pearson",
	"measure/thingsvision/rsa-rdm=gaussian-compare=spearman",
	"measure/xrsa_awes/feature_space_linear_cka",
	"measure/xrsa_awes/feature_space_linear_cka-debiased_True",
	"measure/yuanli2333",
	"measure/yuanli2333/cka",
	"netrep/LinearCKA",
	"netrep/LinearMetric_angular",
	"netrep/LinearMetric_euclidean",
	"netrep/PermutationMetric_angular",
	"netrep/PermutationMetric_euclidean",
	"nn_similarity_index/bures_distance",
	"nn_similarity_index/cka",
	"nn_similarity_index/euclidean",
	"nn_similarity_index/nbs",
	"nnsrm_neurips18/isc",
	"nnsrm_neurips18/procrustes",
	"nnsrm_neurips18/rsa",
	"paper.barannikov2022",
	"paper.bhatia2017",
	"paper.boixadsera2022",
	"paper.camastra2016",
	"paper.conwell2023",
	"paper.ding2021",
	"paper.gwilliam2022",
	"paper.hamilton2016a",
	"paper.hamilton2016b",
	"paper.hryniowski2020",
	"paper.khrulkov2018",
	"paper.klabunde2023",
	"paper.kornblith2019",
	"paper.kriegeskorte2008",
	"paper.lange2022",
	"paper.lange2023",
	"paper.li2016",
	"paper.lu2022",
	"paper.may2019",
	"paper.morcos2018",
	"paper.raghu2017",
	"paper.schrimpf2018",
	"paper.schrimpf2020",
	"paper.schumacher2021",
	"paper.shahbazi2021",
	"paper.szekely2007",
	"paper.tang2020",
	"paper.tsitsulin2020",
	"paper.wang2018",
	"paper.wang2020",
	"paper.wang2021",
	"paper.wang2022",
	"paper.williams2021",
	"paper.yanai1974",
	"paper.yin2018",
	"paper/platonic",
	"paper/representation_similarity",
	"paper/sim_metric",
	"paper/svcca",
	"platonic/cka",
	"platonic/cka_rbf",
	"platonic/cknna_topk",
	"platonic/cycle_knn_topk",
	"platonic/edit_distance_knn_topk",
	"platonic/lcs_knn_topk",
	"platonic/mutual_knn_topk",
	"platonic/svcca",
	"platonic/unbiased_cka",
	"platonic/unbiased_cka_rbf",
	"postprocessing/angular_to_euclidean_shape_metric",
	"postprocessing/arccos",
	"postprocessing/cos",
	"postprocessing/euclidean_to_angular_shape_metric",
	"postprocessing/mean_score",
	"postprocessing/normalize_pi_half",
	"postprocessing/one_minus",
	"postprocessing/sqrt",
	"postprocessing/square",
	"postprocessing/tensor_to_float",
	"preprocessing/array_to_tensor",
	"preprocessing/center_columns",
	"preprocessing/center_rows",
	"preprocessing/center_rows_columns",
	"preprocessing/pca-dim10",
	"preprocessing/pca-var95",
	"preprocessing/pca-var99",
	"preprocessing/reshape2d",
	"preprocessing/transpose",
	"preprocessing/zero_padding",
	"rdm/contrasim/euclidean",
	"rdm/contrasim/euclidean_normalized",
	"rdm/contrasim/squared_euclidean",
	"rdm/contrasim/squared_euclidean_normalized",
	"rdm/correcting_cka_alignment/euclidean",
	"rdm/correcting_cka_alignment/euclidean_normalized",
	"rdm/correcting_cka_alignment/squared_euclidean",
	"rdm/correcting_cka_alignment/squared_euclidean_normalized",
	"rdm/netrep/euclidean",
	"rdm/netrep/euclidean_normalized",
	"rdm/netrep/squared_euclidean",
	"rdm/netrep/squared_euclidean_normalized",
	"rdm/representation_similarity/euclidean",
	"rdm/representation_similarity/euclidean_normalized",
	"rdm/representation_similarity/squared_euclidean",
	"rdm/representation_similarity/squared_euclidean_normalized",
	"rdm/rsatoolbox/euclidean",
	"rdm/rsatoolbox/euclidean_normalized",
	"rdm/rsatoolbox/squared_euclidean",
	"rdm/rsatoolbox/squared_euclidean_normalized",
	"rdm/rsatoolbox/squared_mahalanobis",
	"rdm/rsatoolbox/squared_mahalanobis_normalized",
	"rdm/thingsvision/correlation",
	"rdm/thingsvision/correlation_normalized",
	"rdm/thingsvision/cosine",
	"rdm/thingsvision/cosine_normalized",
	"rdm/thingsvision/euclidean",
	"rdm/thingsvision/euclidean_normalized",
	"rdm/thingsvision/gaussian",
	"rdm/thingsvision/gaussian_normalized",
	"rdm/thingsvision/squared_euclidean",
	"rdm/thingsvision/squared_euclidean_normalized",
	"representation_similarity/cca",
	"representation_similarity/cka",
	"representation_similarity/cka_debiased",
	"repsim/AngularCKA.Laplace[{sigma}]",
	"repsim/AngularCKA.SqExp[{sigma}]",
	"repsim/AngularCKA.linear",
	"repsim/AngularCKA.unb.Laplace[{sigma}]",
	"repsim/AngularCKA.unb.SqExp[{sigma}]",
	"repsim/AngularCKA.unb.linear",
	"repsim/ShapeMetric[{alpha}][angular]",
	"repsim/ShapeMetric[{alpha}][euclidean]",
	"resi/AlignedCosineSimilarity",
	"resi/CKA",
	"resi/ConcentricityDifference",
	"resi/DistanceCorrelation",
	"resi/EigenspaceOverlapScore",
	"resi/Gulp",
	"resi/HardCorrelationMatch",
	"resi/IMDScore",
	"resi/JaccardSimilarity",
	"resi/LinearRegression",
	"resi/MagnitudeDifference",
	"resi/OrthogonalAngularShapeMetricCentered",
	"resi/OrthogonalProcrustesCenteredAndNormalized",
	"resi/PWCCA",
	"resi/PermutationProcrustes",
	"resi/ProcrustesSizeAndShapeDistance",
	"resi/RSA_correlation_euclidean",
	"resi/RSA_correlation_spearman",
	"resi/RSA_euclidean_euclidean",
	"resi/RSA_euclidean_spearman",
	"resi/RSMNormDifference",
	"resi/RankSimilarity",
	"resi/SVCCA",
	"resi/SecondOrderCosineSimilarity",
	"resi/SoftCorrelationMatch",
	"resi/UniformityDifference",
	"rsatoolbox/rsa-correlation-bures",
	"rsatoolbox/rsa-correlation-bures_metric",
	"rsatoolbox/rsa-correlation-corr",
	"rsatoolbox/rsa-correlation-corr_cov",
	"rsatoolbox/rsa-correlation-cosine",
	"rsatoolbox/rsa-correlation-cosine_cov",
	"rsatoolbox/rsa-correlation-kendall",
	"rsatoolbox/rsa-correlation-rho_a",
	"rsatoolbox/rsa-correlation-spearman",
	"rsatoolbox/rsa-correlation-tau_a",
	"rsatoolbox/rsa-correlation-tau_b",
	"rsatoolbox/rsa-euclidean-bures",
	"rsatoolbox/rsa-euclidean-bures_metric",
	"rsatoolbox/rsa-euclidean-corr",
	"rsatoolbox/rsa-euclidean-corr_cov",
	"rsatoolbox/rsa-euclidean-cosine",
	"rsatoolbox/rsa-euclidean-cosine_cov",
	"rsatoolbox/rsa-euclidean-kendall",
	"rsatoolbox/rsa-euclidean-rho_a",
	"rsatoolbox/rsa-euclidean-spearman",
	"rsatoolbox/rsa-euclidean-tau_a",
	"rsatoolbox/rsa-euclidean-tau_b",
	"rsatoolbox/rsa-mahalanobis-bures",
	"rsatoolbox/rsa-mahalanobis-bures_metric",
	"rsatoolbox/rsa-mahalanobis-corr",
	"rsatoolbox/rsa-mahalanobis-corr_cov",
	"rsatoolbox/rsa-mahalanobis-cosine",
	"rsatoolbox/rsa-mahalanobis-cosine_cov",
	"rsatoolbox/rsa-mahalanobis-kendall",
	"rsatoolbox/rsa-mahalanobis-rho_a",
	"rsatoolbox/rsa-mahalanobis-spearman",
	"rsatoolbox/rsa-mahalanobis-tau_a",
	"rsatoolbox/rsa-mahalanobis-tau_b",
	"rsatoolbox/rsa-poisson-bures",
	"rsatoolbox/rsa-poisson-bures_metric",
	"rsatoolbox/rsa-poisson-corr",
	"rsatoolbox/rsa-poisson-corr_cov",
	"rsatoolbox/rsa-poisson-cosine",
	"rsatoolbox/rsa-poisson-cosine_cov",
	"rsatoolbox/rsa-poisson-kendall",
	"rsatoolbox/rsa-poisson-rho_a",
	"rsatoolbox/rsa-poisson-spearman",
	"rsatoolbox/rsa-poisson-tau_b",
	"sim_metric/lin_cka_dist",
	"sim_metric/mean_cca_corr",
	"sim_metric/mean_sq_cca_corr",
	"sim_metric/procrustes",
	"sim_metric/pwcca_dist",
	"similarity/representation_similarity/centered-cosine",
	"similarity/rsatoolbox/zero_diagonal-bures",
	"svcca/cca",
	"svcca/cca_squared_correlation",
	"svcca/pls",
	"svcca/pwcca",
	"thingsvision/cka_kernel_linear_biased",
	"thingsvision/cka_kernel_linear_unbiased",
	"thingsvision/cka_kernel_rbf_biased_sigma_1.0",
	"thingsvision/cka_kernel_rbf_unbiased_sigma_1.0",
	"thingsvision/rsa_method_correlation_corr_method_pearson",
	"thingsvision/rsa_method_correlation_corr_method_spearman",
	"thingsvision/rsa_method_cosine_corr_method_pearson",
	"thingsvision/rsa_method_cosine_corr_method_spearman",
	"thingsvision/rsa_method_euclidean_corr_method_pearson",
	"thingsvision/rsa_method_euclidean_corr_method_spearman",
	"thingsvision/rsa_method_gaussian_corr_method_pearson",
	"thingsvision/rsa_method_gaussian_corr_method_spearman"
]


