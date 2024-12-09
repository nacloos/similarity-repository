import os
import pickle

import numpy as np

from mouse_vision.models.model_layers import MODEL_LAYERS, MODEL_LAYERS_EXTENDED
from mouse_vision.neural_mappers.utils import generate_train_test_img_splits
from mouse_vision.reliability.spec_map_utils import generate_spec_map
from mouse_vision.reliability.metrics import noise_estimation
from mouse_vision.core.utils import open_dataset
from mouse_vision.core.utils import get_params_from_workernum
from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import (
    MODEL_FEATURES_SAVE_DIR,
    MODEL_FEATURES_EXTENDED_SAVE_DIR,
    NEURAL_FIT_RESULTS_DIR_NEW,
    NEURAL_FIT_RESULTS_DIR_EXTENDED_NEW,
)

# 408
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
    "vgg16_ir_64x64",
    "resnet18_ir_64x64",
    "resnet34_ir_64x64",
    "resnet50_ir_64x64",
    "resnet101_ir_64x64",
    "resnet152_ir_64x64",
]

# 258
PRIMATE_64_MODELS = [
    "alexnet_64x64_input_pool_6",
    "resnet18_64x64_input",
    "vgg16_64x64_input",
    "alexnet_bn_ir_64x64_input_pool_6",
    "resnet18_ir_64x64",
]

# 264
SHI_MODELS = ["shi_mousenet", "shi_mousenet_vispor5"]

# 180
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
    "untrained_simplified_mousenet_single_stream",
]

# 300
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
    "untrained_simplified_mousenet_dual_stream_visp_3x3_bn",
]

# 540
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
    "untrained_simplified_mousenet_six_stream_visp_3x3_bn",
]

# 31
IR_224_UNTRAINED_ALEXNET64_MODELS = [
    "alexnet_ir_224x224",
    "simplified_mousenet_single_stream_ir_224x224",
    "simplified_mousenet_dual_stream_visp_3x3_ir_224x224",
    "simplified_mousenet_six_stream_visp_3x3_ir_224x224",
    "untrained_alexnet_64x64_input_pool_6",
]

SHORT_ALEXNET_MODELS = [
    "alexnet_two_64x64",
    "alexnet_three_64x64",
    "alexnet_four_64x64",
    "alexnet_five_64x64",
    "alexnet_six_64x64",
]

DEEPER_RESNET_64_MODELS = [
    "resnet34_64x64_input",
    "resnet50_64x64_input",
    "resnet101_64x64_input",
    "resnet152_64x64_input",
]

# Already included in ALL_MODELS
DEEPER_RESNET_IR_64_MODELS = [
    "resnet34_ir_64x64",
    "resnet50_ir_64x64",
    "resnet101_ir_64x64",
    "resnet152_ir_64x64",
]

RL_END_TO_END_MODELS = ["alexnet_64x64_rl_scratch_truncated"]

IR_EGO_MAZE_MODELS = ["alexnet_ir_dmlocomotion"]

OTHER_CONTRASTIVE_ALEXNET_MODELS = ["alexnet_bn_simsiam_64x64", "alexnet_bn_simclr_64x64", "alexnet_bn_mocov2_64x64"]

SUP_ALEXNET_IR_TRANSFORM_MODELS = ["alexnet_bn_64x64_input_pool_6_with_ir_transforms", "alexnet_64x64_input_pool_6_with_ir_transforms"]

CPC_MONKEYNET_2P_MODELS = ["untrained_monkeynet_2p_cpc", "monkeynet_2p_cpc"]

ALL_MODELS = (
    PRIMATE_MODELS
    + UNSUPERVISED_PRIMATE_MODELS
    + SHI_MODELS
    + SINGLE_STREAM_MODELS
    + DUAL_STREAM_MODELS
    + SIX_STREAM_MODELS
    + IR_224_UNTRAINED_ALEXNET64_MODELS
    + SHORT_ALEXNET_MODELS
    + DEEPER_RESNET_64_MODELS
    + RL_END_TO_END_MODELS
    + IR_EGO_MAZE_MODELS
    + SUP_ALEXNET_IR_TRANSFORM_MODELS
    + CPC_MONKEYNET_2P_MODELS
)

MODEL_REGISTRY = {
    "primate_models": PRIMATE_MODELS + UNSUPERVISED_PRIMATE_MODELS,
    "short_alexnet_models": SHORT_ALEXNET_MODELS,
    "deeper_resnet_64_models": DEEPER_RESNET_64_MODELS,
    "deeper_resnet_ir_64_models": DEEPER_RESNET_IR_64_MODELS,
    "primate_64_models": PRIMATE_64_MODELS,
    "shi_models": SHI_MODELS,
    "single_stream_models": SINGLE_STREAM_MODELS,
    "dual_stream_models": DUAL_STREAM_MODELS,
    "six_stream_models": SIX_STREAM_MODELS,
    "ir_224_untrained_alex64": IR_224_UNTRAINED_ALEXNET64_MODELS,
    "alexnet_224": ["alexnet"],
    "cpc_monkeynet_2p": CPC_MONKEYNET_2P_MODELS,
    "all_models": ALL_MODELS,
    "rl_end_to_end_models": RL_END_TO_END_MODELS,
    "ir_ego_models": IR_EGO_MAZE_MODELS,
    "other_conunsup_alexnet": OTHER_CONTRASTIVE_ALEXNET_MODELS,
    "sup_alexnet_ir_64": SUP_ALEXNET_IR_TRANSFORM_MODELS,
    "vgg16": ["vgg16"],
}

for m in ALL_MODELS:
    if (m not in MODEL_REGISTRY.keys()) and ([m] not in MODEL_REGISTRY.values()):
        MODEL_REGISTRY[m] = [m]


def _obtain_shuffled_features(all_model_features, desired_model_layer, concat_model_feats):
    # all_model_features is a dictionary where each key is the model layer and the value
    # is the layer features
    assert desired_model_layer in all_model_features.keys()
    curr_layer_features = all_model_features[desired_model_layer]
    num_desired_features = curr_layer_features.shape[1]

    print(f"All features shape: {concat_model_feats.shape}")

    # Obtain subset of features
    idxs = np.random.permutation(concat_model_feats.shape[1])
    subset = idxs[:num_desired_features]
    desired_feats = concat_model_feats[:, subset]

    print(f"Desired features for model layer {desired_model_layer}: {desired_feats.shape}")
    return desired_feats


def _concat_all_model_features(model_features_dict):
    # Concatenate features from all model layers
    all_concat_feats = None
    for layer_name in model_features_dict.keys():
        curr_feats = model_features_dict[layer_name]
        assert curr_feats.ndim == 2
        if all_concat_feats is None:
            all_concat_feats = curr_feats
        else:
            all_concat_feats = np.concatenate((all_concat_feats, curr_feats), axis=1)
    return all_concat_feats


def build_param_lookup(
    models=None,
    dataset_name="neuropixels",
    map_kwargs=None,
    num_train_test_splits=10,
    train_frac=0.5,
    metric=None,
    correction="spearman_brown_split_half_denominator",
    splithalf_r_thresh=None,
    do_extended=False,
    use_64px_input=False,
    mds_metric_shuffle=False,
):
    print(f"Extended model: {do_extended}")
    print(f"Using 64 px input features: {use_64px_input}")
    print(f"Doing MDS metric shuffle control: {mds_metric_shuffle}")

    assert models is not None
    assert metric is not None
    assert map_kwargs is not None
    assert "map_type" in map_kwargs.keys()
    map_type = map_kwargs["map_type"]

    # Generate basic information per specimen and across them
    if splithalf_r_thresh is not None:
        visual_area_spec_map = generate_spec_map(
            dataset_name=dataset_name,
            equalize_source_units=False,
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
    for v in VISUAL_AREAS:

        # Loop through each model
        for model_name in models:

            # arch_name is the based architecture name of an untrained model
            if "untrained" == model_name.split("_")[0]:
                arch_name = model_name[len("untrained") + 1 :]
            else:
                arch_name = model_name

            # Loop through each model layer in each model
            if do_extended:
                all_layers = MODEL_LAYERS_EXTENDED[arch_name]
            else:
                all_layers = MODEL_LAYERS[arch_name]

            _concat_model_features = None  # For MDS metric shuffle control
            for model_layer in all_layers:

                # Set up results directory and results file name. model_name includes
                # 'untrained' if that was the desired model, unlike arch_name.
                if do_extended:
                    results_base_dir = NEURAL_FIT_RESULTS_DIR_EXTENDED_NEW
                else:
                    results_base_dir = NEURAL_FIT_RESULTS_DIR_NEW

                # If use 64 px inputs, then store in a subdirectory
                if use_64px_input:
                    results_base_dir = os.path.join(results_base_dir, "64px_features")

                # If doing the MDS metric control where units are shuffled across layers
                if mds_metric_shuffle:
                    results_base_dir = os.path.join(results_base_dir, "mds_metric_shuffle")

                if splithalf_r_thresh is not None:
                    results_dir = os.path.join(
                        results_base_dir,
                        f"{dataset_name}sphr{splithalf_r_thresh}/{model_name}/{v}/{map_type}",
                    )
                else:
                    results_dir = os.path.join(
                        results_base_dir,
                        f"{dataset_name}/{model_name}/{v}/{map_type}",
                    )
                if not os.path.exists(results_dir):
                    os.makedirs(results_dir)

                # Obtain model layer features. model_name is the architecture that
                # was trained or not.
                if do_extended:
                    model_features_path = os.path.join(
                        MODEL_FEATURES_EXTENDED_SAVE_DIR, dataset_name
                    )
                else:
                    model_features_path = os.path.join(
                        MODEL_FEATURES_SAVE_DIR, dataset_name
                    )

                # If we want to test models on 64 px inputs regardless of the resolution
                # at which the model was trained, use those features instead.
                if use_64px_input:
                    model_features_path = os.path.join(
                        model_features_path, "64px_features", f"{model_name}.pkl"
                    )
                else:
                    model_features_path = os.path.join(
                        model_features_path, f"{model_name}.pkl"
                    )

                model_features = open_dataset(model_features_path)
                assert model_layer in model_features.keys()

                # MDS metric control: shuffle units across all model layers
                if mds_metric_shuffle:
                    # Concatenate all model features if not already done so
                    if _concat_model_features is None:
                        _concat_model_features = _concat_all_model_features(model_features)

                    # Obtain random subset of all model features for current model layer
                    model_features = _obtain_shuffled_features(
                        model_features, model_layer, _concat_model_features
                    )
                else:
                    model_features = model_features[model_layer]

                # Now set up parameters for the main neural fitting function below.
                neural_fit_kwargs = {
                    "model_layer_name": model_layer,
                    "model_features": model_features,
                    "visual_area_spec_map": visual_area_spec_map[v],
                    "map_kwargs": map_kwargs,
                    "num_train_test_splits": num_train_test_splits,
                    "train_frac": train_frac,
                    "metric": metric,
                    "correction": correction,
                    "results_dir": results_dir,
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
    model_layer_name,
    model_features,
    visual_area_spec_map,
    map_kwargs,
    num_train_test_splits=10,
    train_frac=0.5,
    metric=None,
    correction="spearman_brown_split_half_denominator",
    results_dir=None,
):
    assert results_dir is not None
    assert os.path.isdir(results_dir)
    model_layer_name = model_layer_name.replace("/", "_")  # VISp2/3 -> VISp2_3
    results_fname = os.path.join(results_dir, f"{model_layer_name}.pkl")

    train_test_splits = generate_train_test_img_splits(
        num_splits=num_train_test_splits, train_frac=train_frac
    )

    # Perform neural fits per specimen
    results_across_specs = dict()
    specimens = [k for k in visual_area_spec_map.keys() if isinstance(k, np.int64)]
    for specimen in specimens:
        print(f"Fitting for specimen {specimen}")
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
                n_jobs=10,
            )

            curr_results = np.expand_dims(curr_results, axis=0)
            spec_results.append(curr_results)

        # spec_results: (num_train_test_splits, num_bootstrap_split_half, num_targets)
        spec_results = np.concatenate(spec_results, axis=0)
        # commenting out for RSA, as the end user we will know the shape anyway
        # assert spec_results.ndim == 3
        results_across_specs[specimen] = spec_results

    pickle.dump({"scores": results_across_specs}, open(results_fname, "wb"))


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset", type=str, default="neuropixels", choices=["neuropixels", "calcium"]
    )
    parser.add_argument("--num-train-test-splits", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--map-type", type=str, default="pls")
    parser.add_argument("--model-group", type=str, default="single_stream_models")
    parser.add_argument("--do-extended", type=bool, default=False)
    parser.add_argument("--count-only", type=bool, default=False)
    parser.add_argument("--use-64px-input", type=bool, default=False)
    parser.add_argument("--splithalf-r-thresh", type=float, default=None)
    parser.add_argument("--mds-metric-shuffle-units", type=bool, default=False)

    args = parser.parse_args()

    # Set up which models to do neural fits on
    assert args.model_group in MODEL_REGISTRY.keys()
    models = MODEL_REGISTRY[args.model_group]

    print(f"Fitting {models}")

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
    elif args.map_type == "corr":
        metric = "pearsonr"
        map_kwargs = {
            "map_type": "corr",
            "map_kwargs": {"identity": True, "percentile": 100},
        }
    else:
        assert 0, f"{args.map_type} not supported yet."

    print("Looking up params...")
    param_lookup = build_param_lookup(
        models=models,
        dataset_name=args.dataset,
        map_kwargs=map_kwargs,
        num_train_test_splits=args.num_train_test_splits,
        train_frac=args.train_frac,
        metric=metric,
        correction="spearman_brown_split_half_denominator",
        splithalf_r_thresh=args.splithalf_r_thresh,
        do_extended=args.do_extended,
        use_64px_input=args.use_64px_input,
        mds_metric_shuffle=args.mds_metric_shuffle_units,
    )

    print("TOTAL NUMBER OF JOBS: {}".format(len(list(param_lookup.keys()))))
    if not args.count_only:
        curr_params = get_params_from_workernum(
            worker_num=os.environ.get("SLURM_ARRAY_TASK_ID"), param_lookup=param_lookup
        )
        print("Current params", curr_params)
        perform_neural_fits(**curr_params)
