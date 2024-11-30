import os
import pickle

import numpy as np

from mouse_vision.models.model_layers import MODEL_LAYERS
from mouse_vision.neural_mappers.utils import generate_train_test_img_splits
from mouse_vision.reliability.spec_map_utils import generate_spec_map
from mouse_vision.reliability.metrics import noise_estimation
from mouse_vision.core.utils import open_dataset
from mouse_vision.core.utils import get_params_from_workernum
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import (
    MODEL_FEATURES_SAVE_DIR,
    NEURAL_FIT_RESULTS_DIR_NEW,
)

PRIMATE_MODELS = [
    "alexnet",
    "alexnet_64x64_input_pool_6",
    "resnet18",
    "resnet18_64x64_input",
    "vgg16",
    "vgg16_64x64_input",
]

UNSUPERVISED_PRIMATE_MODELS = [
    "alexnet_bn_ir_64x64_input_pool_6",
    "resnet18_ir_64x64",
    "alexnet_ir_224x224",
    "untrained_alexnet_64x64_input_pool_6"
]

SHI_MODELS = ["shi_mousenet", "shi_mousenet_vispor5"]

PRIMATE_MODELS_64 = [
    "alexnet_64x64_input_pool_6",
    "resnet18_64x64_input",
    "vgg16_64x64_input",
    "alexnet_bn_ir_64x64_input_pool_6",
    "resnet18_ir_64x64",
]

SINGLE_STREAM_MODELS = [
    "simplified_mousenet_single_stream_cifar10",
    "simplified_mousenet_single_stream",
    "simplified_mousenet_single_stream_rotnet",
    "simplified_mousenet_single_stream_ir",
    "simplified_mousenet_single_stream_simsiam",
    "simplified_mousenet_single_stream_simclr",
    "simplified_mousenet_single_stream_mocov2",
    "simplified_mousenet_ae_single_stream",
    "simplified_mousenet_depth_hour_glass_single_stream",
    "simplified_mousenet_single_stream_ir_224x224",
    "untrained_simplified_mousenet_single_stream",
]

DUAL_STREAM_MODELS = [
    "simplified_mousenet_dual_stream_visp_3x3_bn_cifar10",
    "simplified_mousenet_dual_stream_visp_3x3_bn",
    "simplified_mousenet_dual_stream_visp_3x3_rotnet",
    "simplified_mousenet_dual_stream_visp_3x3_ir",
    "simplified_mousenet_dual_stream_visp_3x3_simsiam",
    "simplified_mousenet_dual_stream_visp_3x3_simclr",
    "simplified_mousenet_dual_stream_visp_3x3_mocov2",
    "simplified_mousenet_ae_dual_stream",
    "simplified_mousenet_depth_hour_glass_dual_stream",
    "simplified_mousenet_dual_stream_visp_3x3_ir_224x224",
    "untrained_simplified_mousenet_dual_stream_visp_3x3_bn",
]

SIX_STREAM_MODELS = [
    "simplified_mousenet_six_stream_visp_3x3_bn_cifar10",
    "simplified_mousenet_six_stream_visp_3x3_bn",
    "simplified_mousenet_six_stream_visp_3x3_rotnet",
    "simplified_mousenet_six_stream_visp_3x3_ir",
    "simplified_mousenet_six_stream_visp_3x3_simsiam",
    "simplified_mousenet_six_stream_visp_3x3_simclr",
    "simplified_mousenet_six_stream_visp_3x3_mocov2",
    "simplified_mousenet_ae_six_stream",
    "simplified_mousenet_depth_hour_glass_six_stream",
    "simplified_mousenet_six_stream_visp_3x3_ir_224x224",
    "untrained_simplified_mousenet_six_stream_visp_3x3_bn",
]

CPC_MONKEYNET_2P_MODELS = ["untrained_monkeynet_2p_cpc", "monkeynet_2p_cpc"]

INTERMEDIATE_RES_MODELS = ["simplified_mousenet_dual_stream_visp_3x3_ir_32x32", "simplified_mousenet_dual_stream_visp_3x3_ir_44x44", \
                         "simplified_mousenet_dual_stream_visp_3x3_ir_84x84", "simplified_mousenet_dual_stream_visp_3x3_ir_104x104", \
                         "simplified_mousenet_dual_stream_visp_3x3_ir_124x124", "simplified_mousenet_dual_stream_visp_3x3_ir_144x144", \
                         "simplified_mousenet_dual_stream_visp_3x3_ir_164x164", "simplified_mousenet_dual_stream_visp_3x3_ir_184x184", \
                         "simplified_mousenet_dual_stream_visp_3x3_ir_204x204", \
                         "alexnet_ir_84x84", "alexnet_ir_104x104", "alexnet_ir_124x124", "alexnet_ir_144x144", "alexnet_ir_164x164", \
                         "alexnet_ir_184x184", "alexnet_ir_204x204"]

SHORT_ALEXNET_MODELS = [
    "alexnet_two_64x64",
    "alexnet_three_64x64",
    "alexnet_four_64x64",
    "alexnet_five_64x64",
    "alexnet_six_64x64",
]

NEW_UNSUPERVISED_PRIMATE_MODELS = [
    "vgg16_ir_64x64",
    "resnet34_ir_64x64",
    "resnet50_ir_64x64",
    "resnet101_ir_64x64",
    "resnet152_ir_64x64",
]

ALL_MODELS = (
    PRIMATE_MODELS
    + UNSUPERVISED_PRIMATE_MODELS
    + NEW_UNSUPERVISED_PRIMATE_MODELS
    + SHI_MODELS
    + SINGLE_STREAM_MODELS
    + DUAL_STREAM_MODELS
    + SIX_STREAM_MODELS
    + INTERMEDIATE_RES_MODELS
    + SHORT_ALEXNET_MODELS
    + CPC_MONKEYNET_2P_MODELS
)

MODEL_REGISTRY = {
    "primate_models": PRIMATE_MODELS + UNSUPERVISED_PRIMATE_MODELS,
    "new_unsup_primate_models": NEW_UNSUPERVISED_PRIMATE_MODELS,
    "short_alexnet_models": SHORT_ALEXNET_MODELS,
    "shi_models": SHI_MODELS,
    "shi_mousenet": ["shi_mousenet"],
    "shi_mousenet_vispor5": ["shi_mousenet_vispor5"],
    "single_stream_models": SINGLE_STREAM_MODELS,
    "dual_stream_models": DUAL_STREAM_MODELS,
    "six_stream_models": SIX_STREAM_MODELS,
    "primate_models_64": PRIMATE_MODELS_64,
    "alexnet_full": ["alexnet"],
    "resnet18_full": ["resnet18"],
    "vgg16_full": ["vgg16"],
    "cpc_monkeynet_2p": CPC_MONKEYNET_2P_MODELS,
    "alexnet_ir_224": ["alexnet_ir_224x224"],
    "six_ir_224": ["simplified_mousenet_six_stream_visp_3x3_ir_224x224"],
    "dual_ir_224": ["simplified_mousenet_dual_stream_visp_3x3_ir_224x224"],
    "single_ir_224": ["simplified_mousenet_single_stream_ir_224x224"],
    "untrained_alexnet_64": ["untrained_alexnet_64x64_input_pool_6"],
    "intermediate_res": INTERMEDIATE_RES_MODELS,
}

for m in ALL_MODELS:
    if (m not in MODEL_REGISTRY.keys()) and ([m] not in MODEL_REGISTRY.values()):
        MODEL_REGISTRY[m] = [m]

def build_param_lookup(
    visual_areas=None,
    models=None,
    dataset_name="neuropixels",
    map_kwargs=None,
    num_train_test_splits=10,
    train_frac=0.5,
    metric=None,
    correction="spearman_brown_split_half_denominator",
    job_file_name="incomplete_jobs.pkl",
    splithalf_r_thresh=None,
    n_jobs=10,
):

    if visual_areas is None:
        visual_areas = VISUAL_AREAS
    else:
        for v in visual_areas:
            assert(v in VISUAL_AREAS)

    incomplete_jobs = open_dataset(os.path.join(NEURAL_FIT_RESULTS_DIR_NEW, job_file_name))[dataset_name]

    assert models is not None
    assert metric is not None
    assert map_kwargs is not None
    assert "map_type" in map_kwargs.keys()
    map_type = map_kwargs["map_type"]

    # Generate basic information per specimen and across them
    if splithalf_r_thresh is not None:
        visual_area_spec_map = generate_spec_map(
            dataset_name=dataset_name, equalize_source_units=False,
            splithalf_r_thresh=splithalf_r_thresh,
        )
    else:
        visual_area_spec_map = generate_spec_map(
            dataset_name=dataset_name, equalize_source_units=False
        )

    # Build param lookup. Here, one job is associated with one visual area,
    # one model, and one layer. Each job will perform neural fits for all
    # specimens in the visual area.
    param_lookup = {}
    key = 0

    # Loop through each visual area
    for v in visual_areas:

        specimens = [k for k in visual_area_spec_map[v].keys() if isinstance(k, np.int64)]

        # Loop through each model
        for model_name in models:

            if model_name not in incomplete_jobs.keys():
                continue

            # arch_name is the based architecture name of an untrained model
            if "untrained" == model_name.split("_")[0]:
                arch_name = model_name[len("untrained") + 1 :]
            else:
                arch_name = model_name

            # Loop through each model layer in each model
            has_incomplete_specimens = False
            if isinstance(incomplete_jobs[model_name][v], list):
                incomplete_model_layers = incomplete_jobs[model_name][v]
            else:
                has_incomplete_specimens = True
                incomplete_model_layers = list(incomplete_jobs[model_name][v].keys())

            for model_layer in incomplete_model_layers:

                if has_incomplete_specimens:
                    incomplete_specimens = incomplete_jobs[model_name][v][model_layer]
                else:
                    incomplete_specimens = specimens

                for specimen in incomplete_specimens:

                    # Set up results directory and results file name. model_name includes
                    # 'untrained' if that was the desired model, unlike arch_name.
                    model_layer_name = model_layer.replace("/", "_") # VISp2/3 -> VISp2_3
                    if splithalf_r_thresh is not None:
                        results_dir = os.path.join(
                            NEURAL_FIT_RESULTS_DIR_NEW,
                            f"per_specimen/{dataset_name}sphr{splithalf_r_thresh}/{model_name}/{v}/{map_type}/{model_layer_name}",
                        )
                    else:
                        results_dir = os.path.join(
                            NEURAL_FIT_RESULTS_DIR_NEW,
                            f"per_specimen/{dataset_name}/{model_name}/{v}/{map_type}/{model_layer_name}",
                        )
                    if not os.path.exists(results_dir):
                        os.makedirs(results_dir)

                    # Now set up parameters for the main neural fitting function below.
                    neural_fit_kwargs = {
                        "model_layer": model_layer,
                        "model_name": model_name,
                        "dataset_name": dataset_name,
                        "visual_area_spec_map": visual_area_spec_map[v],
                        "specimen": specimen,
                        "map_kwargs": map_kwargs,
                        "num_train_test_splits": num_train_test_splits,
                        "train_frac": train_frac,
                        "metric": metric,
                        "correction": correction,
                        "results_dir": results_dir,
                        "n_jobs": n_jobs,
                    }

                    param_lookup[str(key)] = neural_fit_kwargs
                    key += 1

    return param_lookup


def _check_dim(spec_response):
    assert spec_response.ndim == 3
    assert spec_response.dims[0] == "trials"
    assert spec_response.dims[1] == "frame_id"
    assert spec_response.dims[2] == "units"


def perform_neural_fits(
    model_layer,
    model_name,
    dataset_name,
    visual_area_spec_map,
    map_kwargs,
    specimen,
    num_train_test_splits=10,
    train_frac=0.5,
    metric=None,
    correction="spearman_brown_split_half_denominator",
    results_dir=None,
    n_jobs=10,
):
    # Obtain model layer features. model_name is the architecture that
    # was trained or not .
    model_features_path = os.path.join(
        MODEL_FEATURES_SAVE_DIR, dataset_name, f"{model_name}.pkl"
    )
    model_features = open_dataset(model_features_path)
    if model_name in ["shi_mousenet", "shi_mousenet_vispor5"]:
        # only do this for the shi mousenet since the features were saved with "/"
        model_layer = model_layer.replace("_", "/")
    assert model_layer in model_features.keys()
    model_features = model_features[model_layer]

    assert results_dir is not None
    assert os.path.isdir(results_dir)
    results_fname = os.path.join(results_dir, f"{specimen}.pkl")

    train_test_splits = generate_train_test_img_splits(
        num_splits=num_train_test_splits, train_frac=train_frac
    )

    # Perform neural fits per specimen
    specimen_responses = visual_area_spec_map[specimen]
    _check_dim(specimen_responses)

    spec_results = list()
    for curr_sp in train_test_splits:
        curr_results = noise_estimation(
            target_N=specimen_responses,
            source_N=model_features,
            source_map_kwargs=map_kwargs,
            train_img_idx=curr_sp["train"],
            test_img_idx=curr_sp["test"],
            metric=metric,
            mode=correction,
            center=np.nanmean,
            parallelize_per_target_unit=False,
            num_source_units=None,
            summary_center="raw",
            sync=True,
            n_ss_iter=None,
            n_iter=100,
            n_jobs=n_jobs,
        )

        curr_results = np.expand_dims(curr_results, axis=0)
        spec_results.append(curr_results)

    # spec_results: (num_train_test_splits, num_bootstrap_split_half, num_targets)
    spec_results = np.concatenate(spec_results, axis=0)

    pickle.dump(spec_results, open(results_fname, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--visual-areas", type=str, default=None)
    parser.add_argument(
        "--dataset", type=str, default="neuropixels", choices=["neuropixels", "calcium"]
    )
    parser.add_argument("--num-train-test-splits", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--map-type", type=str, default="pls")
    parser.add_argument("--model-group", type=str, default="single_stream_models")
    parser.add_argument("--job-file-name", type=str, default="incomplete_jobs.pkl")
    parser.add_argument("--count-only", type=bool, default=False)
    parser.add_argument("--splithalf-r-thresh", type=float, default=None)
    parser.add_argument("--n-jobs", type=int, default=10)

    args = parser.parse_args()

    # Set up which models to do neural fits on
    assert args.model_group in MODEL_REGISTRY.keys()
    models = MODEL_REGISTRY[args.model_group]

    # Set up map kwargs
    if args.map_type == "pls":
        metric = "pearsonr"
        map_kwargs = {
            "map_type": "pls",
            "map_kwargs": {"n_components": 25, "fit_per_target_unit": False},
        }
    elif args.map_type == "identity":
        metric = "rsa"
        map_kwargs = {"map_type": "identity", "map_kwargs": {}}
    else:
        assert 0, f"{args.map_type} not supported yet."

    print("Looking up params...")
    param_lookup = build_param_lookup(
        visual_areas=args.visual_areas.split(",") if args.visual_areas is not None else None,
        models=models,
        dataset_name=args.dataset,
        map_kwargs=map_kwargs,
        num_train_test_splits=args.num_train_test_splits,
        train_frac=args.train_frac,
        metric=metric,
        correction="spearman_brown_split_half_denominator",
        job_file_name=args.job_file_name,
        splithalf_r_thresh=args.splithalf_r_thresh,
        n_jobs=args.n_jobs,
    )

    print("TOTAL NUMBER OF JOBS: {}".format(len(list(param_lookup.keys()))))
    if not args.count_only:
        curr_params = get_params_from_workernum(
            worker_num=os.environ.get("SLURM_ARRAY_TASK_ID"), param_lookup=param_lookup
        )
        print("Current params", curr_params)
        perform_neural_fits(**curr_params)

