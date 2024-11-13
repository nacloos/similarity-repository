from .cca import PWCCA
from .cca import SVCCA
from .cka import CKA
from .correlation_match import HardCorrelationMatch
from .correlation_match import SoftCorrelationMatch
from .distance_correlation import DistanceCorrelation
from .eigenspace_overlap import EigenspaceOverlapScore
from .geometry_score import GeometryScore
from .gulp import Gulp
from .linear_regression import LinearRegression
from .multiscale_intrinsic_distance import IMDScore
from .nearest_neighbor import JaccardSimilarity
from .nearest_neighbor import RankSimilarity
from .nearest_neighbor import SecondOrderCosineSimilarity
from .output_similarity import Disagreement
from .output_similarity import JSD
from .procrustes import AlignedCosineSimilarity
from .procrustes import OrthogonalAngularShapeMetricCentered
from .procrustes import OrthogonalProcrustesCenteredAndNormalized
from .procrustes import PermutationProcrustes
from .procrustes import ProcrustesSizeAndShapeDistance
from .rsa import RSA
from .rsm_norm_difference import RSMNormDifference
from .statistics import ConcentricityDifference
from .statistics import MagnitudeDifference
from .statistics import UniformityDifference
from .output_similarity import AbsoluteAccDiff

CLASSES = [
    PWCCA,
    SVCCA,
    HardCorrelationMatch,
    SoftCorrelationMatch,
    DistanceCorrelation,
    EigenspaceOverlapScore,
    GeometryScore,
    IMDScore,
    Gulp,
    LinearRegression,
    JaccardSimilarity,
    RankSimilarity,
    SecondOrderCosineSimilarity,
    AlignedCosineSimilarity,
    OrthogonalAngularShapeMetricCentered,
    OrthogonalProcrustesCenteredAndNormalized,
    PermutationProcrustes,
    ProcrustesSizeAndShapeDistance,
    RSA,
    RSMNormDifference,
    ConcentricityDifference,
    MagnitudeDifference,
    UniformityDifference,
    CKA,
]

ALL_MEASURES = {m().name: m() for m in CLASSES}

FUNCTIONAL_SIMILARITY_MEASURES = {m().name: m() for m in [JSD, Disagreement]}
