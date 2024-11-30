import numpy as np
import seaborn as sns

# Num Natural Scene Stimuli
NUM_NS_STIM = 118
NUM_NS_STIM_PLUS_GREY = NUM_NS_STIM + 1

# Num trials
NUM_TRIALS = 50

# Time bin constants for binning spike counts and calculating firing rates
TIME_RESOLUTION = 0.01
TIME_BINS = np.arange(0, 0.25 + TIME_RESOLUTION, TIME_RESOLUTION)
# this is the actual number of entries in the time dimension when passing the TIME_BINS array as bin edges to the SDK
# 1 less due to exclusion of double counting of bin edges
NUM_TIMEPOINTS = len(TIME_BINS) - 1

# Desired visual areas
VISUAL_AREAS = ["VISp", "VISl", "VISal", "VISpm", "VISrl", "VISam"]
VISUAL_AREAS_HIER = ["VISp", "VISl", "VISrl", "VISal", "VISpm", "VISam"]

cmap = sns.color_palette("hls", len(VISUAL_AREAS_HIER))
colors = [cmap[i] for i in range(len(VISUAL_AREAS_HIER))]
VISUAL_AREA_COLOR_MAP = {v: colors[v_idx] for v_idx, v in enumerate(VISUAL_AREAS_HIER)}

# Dataset stats
DMLOCOMOTION_NUM_IMGS = 19495116
IMAGENET_NUM_IMGS = 1281167
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

CIFAR10_MEAN = [0.4914, 0.4822, 0.4465]
CIFAR10_STD = [0.2023, 0.1994, 0.2010]

# Neural fit constants
CLASSIC_NOISECORRECTION = {
    "map_type": "identity",
    "num_splits": 1,
    "train_frac": 0.0,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
}

PAIRWISE_INTERANIMAL_NOISECORRECTION = {
    "map_type": "identity",
    "num_splits": 1,
    "train_frac": 0.0,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
    "separate_by_animal": True,
    "interanimal_correction": {
        "n_iter": 900,
        "equalize_source_units": False,
        "map_kwargs": "map_kwargs{}map_typeidentity",
    },
}

HOLDOUT_INTERANIMAL_NOISECORRECTION = {
    "map_type": "identity",
    "num_splits": 1,
    "train_frac": 0.0,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
    "separate_by_animal": True,
    "interanimal_correction": {
        "n_iter": 900,
        "equalize_source_units": False,
        "mode": "holdout",
        "map_kwargs": "map_kwargs{}map_typeidentity",
    },
}

HOLDOUT_INTERANIMAL_NOISECORRECTION_PLS = {
    "map_type": "pls",
    "num_splits": 10,
    "train_frac": 0.5,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
    "separate_by_animal": True,
    "interanimal_correction": {
        "n_iter": 100,
        "equalize_source_units": False,
        "mode": "holdout",
        "map_kwargs": {
            "map_type": "pls",
            "map_kwargs": {"n_components": 25, "fit_per_target_unit": False},
        },
    },
}

HOLDOUT_INTERANIMAL_NOISECORRECTION_CORR = {
    "map_type": "corr",
    "num_splits": 10,
    "train_frac": 0.5,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
    "separate_by_animal": True,
    "interanimal_correction": {
        "n_iter": 100,
        "equalize_source_units": False,
        "mode": "holdout",
        "map_kwargs": {
            "map_type": "corr",
            "map_kwargs": {"identity": True, "percentile": 100},
        },
    },
}

HOLDOUT_INTERANIMAL_NOISECORRECTION_IDENTITY = {
    "map_type": "identity",
    "num_splits": 1,
    "train_frac": 0.0,
    "center_trials": False,
    "mode": "spearman_brown_split_half_denominator",
    "separate_by_animal": True,
    "interanimal_correction": {
        "n_iter": 100,
        "equalize_source_units": False,
        "mode": "holdout",
        "map_kwargs": {"map_type": "identity", "map_kwargs": {}},
    },
}

# SKLEARN TRANSFER CONSTANTS
SVM_CV_C = [5e-2, 5e-1, 1.0, 5e1, 5e2, 5e3, 5e4]
SVM_CV_C_LONG = [
    1e-8,
    5e-8,
    1e-7,
    5e-7,
    1e-6,
    5e-6,
    1e-5,
    5e-5,
    1e-4,
    5e-4,
    1e-3,
    5e-3,
    1e-2,
    5e-2,
    1e-1,
    5e-1,
    1,
    1e1,
    5e1,
    1e2,
    5e2,
    1e3,
    5e3,
    1e4,
    5e4,
    1e5,
    5e5,
    1e6,
    5e6,
    1e7,
    5e7,
    1e8,
    5e8,
]
RIDGE_CV_ALPHA = [0.01, 0.1, 1, 10]
RIDGE_CV_ALPHA_LONG = list(1.0 / np.array(SVM_CV_C_LONG))
