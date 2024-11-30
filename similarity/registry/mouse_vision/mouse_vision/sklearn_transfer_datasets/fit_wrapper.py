import os
import copy

from mouse_vision.core.utils import get_params_from_workernum
from mouse_vision.core.utils import get_base_arch_name
from mouse_vision.core.default_dirs import HVM_SPLIT_FILE, HVM_DATA_PATH, HVM_V3V6_SPLIT_FILE
from mouse_vision.models.model_layers import MODEL_LAYERS
from mouse_vision.sklearn_transfer_datasets import DTDFitter, HvmFitter


# 50
PRIMATE_MODELS = [
    "untrained_alexnet_64x64_input_pool_6",
    "alexnet_64x64_input_pool_6",
    "untrained_resnet18_64x64_input",
    "resnet18_64x64_input",
    "untrained_vgg16_64x64_input",
    "vgg16_64x64_input",
]

# 18
UNSUPERVISED_PRIMATE_MODELS = [
    "alexnet_bn_ir_64x64_input_pool_6",
    "resnet18_ir_64x64",
]

DEEPER_RESNET_MODELS = [
    "resnet34_64x64_input",
    "resnet50_64x64_input",
    "resnet101_64x64_input",
    "resnet152_64x64_input",
]

DEEPER_RESNET_IR_MODELS = [
    "resnet34_ir_64x64",
    "resnet50_ir_64x64",
    "resnet101_ir_64x64",
    "resnet152_ir_64x64",
]

# 44
SHI_MODELS = ["shi_mousenet", "shi_mousenet_vispor5"]

# 32
SINGLE_STREAM_MODELS = [
    "simplified_mousenet_single_stream_cifar10",
    "simplified_mousenet_single_stream",
    "simplified_mousenet_single_stream_rotnet",
    "simplified_mousenet_single_stream_ir",
#    "simplified_mousenet_single_stream_simsiam",
#    "simplified_mousenet_single_stream_simclr",
    "simplified_mousenet_single_stream_mocov2",
    "simplified_mousenet_ae_single_stream",
    "simplified_mousenet_depth_hour_glass_single_stream",
    "untrained_simplified_mousenet_single_stream",
]

# 48
DUAL_STREAM_MODELS = [
    "simplified_mousenet_dual_stream_visp_3x3_bn_cifar10",
    "simplified_mousenet_dual_stream_visp_3x3_bn",
    "simplified_mousenet_dual_stream_visp_3x3_rotnet",
    "simplified_mousenet_dual_stream_visp_3x3_ir",
#    "simplified_mousenet_dual_stream_visp_3x3_simsiam",
#    "simplified_mousenet_dual_stream_visp_3x3_simclr",
    "simplified_mousenet_dual_stream_visp_3x3_mocov2",
    "simplified_mousenet_ae_dual_stream",
    "simplified_mousenet_depth_hour_glass_dual_stream",
    "untrained_simplified_mousenet_dual_stream_visp_3x3_bn",
]

# 80
SIX_STREAM_MODELS = [
    "simplified_mousenet_six_stream_visp_3x3_bn_cifar10",
    "simplified_mousenet_six_stream_visp_3x3_bn",
    "simplified_mousenet_six_stream_visp_3x3_rotnet",
    "simplified_mousenet_six_stream_visp_3x3_ir",
#    "simplified_mousenet_six_stream_visp_3x3_simsiam",
#    "simplified_mousenet_six_stream_visp_3x3_simclr",
    "simplified_mousenet_six_stream_visp_3x3_mocov2",
    "simplified_mousenet_ae_six_stream",
    "simplified_mousenet_depth_hour_glass_six_stream",
    "untrained_simplified_mousenet_six_stream_visp_3x3_bn",
]

# 40
SYNC_BN_MODELS = [
    "simplified_mousenet_single_stream_simsiam",
    "simplified_mousenet_single_stream_simclr",
    "simplified_mousenet_dual_stream_visp_3x3_simsiam",
    "simplified_mousenet_dual_stream_visp_3x3_simclr",
    "simplified_mousenet_six_stream_visp_3x3_simsiam",
    "simplified_mousenet_six_stream_visp_3x3_simclr",
]

RL_MODELS = [
    "alexnet_64x64_rl_scratch_truncated",
    "alexnet_ir_dmlocomotion",
]

ALEXNET_IMNET_MODELS = ["alexnet_bn_ir_64x64_input_pool_6", "alexnet_64x64_input_pool_6"]

# 272
ALL_MODELS = (
    PRIMATE_MODELS
    + UNSUPERVISED_PRIMATE_MODELS
    + SHI_MODELS
    + SINGLE_STREAM_MODELS
    + DUAL_STREAM_MODELS
    + SIX_STREAM_MODELS
    + RL_MODELS
    + DEEPER_RESNET_MODELS
    + DEEPER_RESNET_IR_MODELS
)


MODEL_REGISTRY = {
    "all_models": ALL_MODELS, # excludes syncbn models
    "rl_models": RL_MODELS,
    "alexnet_imnet_models": ALEXNET_IMNET_MODELS,
    "deeper_resnet_models": DEEPER_RESNET_MODELS,
    "deeper_resnet_ir_models": DEEPER_RESNET_IR_MODELS,
    "primate_models": PRIMATE_MODELS + UNSUPERVISED_PRIMATE_MODELS,
    "shi_models": SHI_MODELS,
    "single_stream_models": SINGLE_STREAM_MODELS,
    "dual_stream_models": DUAL_STREAM_MODELS,
    "six_stream_models": SIX_STREAM_MODELS,
    "sync_bn_models": SYNC_BN_MODELS,
}


def build_param_lookup(dataset_name, models, task_type):
    param_lookup = dict()
    key = 0

    for model_name in models:
        arch_name, _ = get_base_arch_name(model_name)

        model_layers = copy.deepcopy(MODEL_LAYERS[arch_name])

        # We also want to perform transfer on the avgpool layer of our simplified
        # mousenets. We have to append "avgpool" here because we do not want to
        # include it in models.model_layers.MODEL_LAYERS, as we do not do neural
        # fits on the "avgpool" layer of our simplified mousenets.
        if "simplified_mousenet" in model_name:
            model_layers.append("avgpool")

        for model_layer in model_layers:
            # Now set up parameters for the main fitting function below.
            if task_type == "8xcat":
                for inner_task_type in ["cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8"]:
                    fit_kwargs = {
                        "dataset_name": dataset_name,
                        "model_name": model_name,
                        "model_layer": model_layer,
                        "task_type": inner_task_type,
                    }

                    param_lookup[str(key)] = fit_kwargs
                    key += 1
            else:
                fit_kwargs = {
                    "dataset_name": dataset_name,
                    "model_name": model_name,
                    "model_layer": model_layer,
                    "task_type": task_type,
                }

                param_lookup[str(key)] = fit_kwargs
                key += 1

    return param_lookup


def perform_transfer_task(dataset_name, model_name, model_layer, task_type):
    if (dataset_name == "hvm_dataset") or (dataset_name == "hvm_v3v6_dataset"):
        assert task_type in ["instance_categorization", "categorization", "cat1", "cat2", "cat3", "cat4", "cat5", "cat6", "cat7", "cat8", "pose", "position", "size"]
        if dataset_name == "hvm_dataset":
            fitter = HvmFitter(model_name=model_name, train_test_splits_file=HVM_SPLIT_FILE, hvm_data_path=HVM_DATA_PATH, name=dataset_name)
        else:
            assert(dataset_name == "hvm_v3v6_dataset")
            fitter = HvmFitter(model_name=model_name, train_test_splits_file=HVM_V3V6_SPLIT_FILE, hvm_data_path=HVM_DATA_PATH, name=dataset_name)
    elif dataset_name == "dtd_dataset":
        assert task_type == "categorization"
        fitter = DTDFitter(model_name=model_name)
    else:
        raise ValueError(f"{dataset_name} not supported.")

    fitter.fit(task_type, model_layer)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset",
        type=str,
        default="hvm_dataset",
        choices=["hvm_dataset", "hvm_v3v6_dataset", "dtd_dataset"],
    )
    parser.add_argument("--model-group", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    parser.add_argument("--task-type", type=str, default="categorization")
    parser.add_argument("--count-only", type=bool, default=False)

    args = parser.parse_args()

    print("Looking up params...")
    if args.model_group is not None:
        models = MODEL_REGISTRY[args.model_group]
    else:
        assert args.model_name is not None
        models = [args.model_name]

    param_lookup = build_param_lookup(args.dataset, models, args.task_type)

    print("TOTAL NUMBER OF JOBS: {}".format(len(list(param_lookup.keys()))))
    if not args.count_only:
        curr_params = get_params_from_workernum(
            worker_num=os.environ.get("SLURM_ARRAY_TASK_ID"), param_lookup=param_lookup
        )
        print("Current params", curr_params)
        perform_transfer_task(**curr_params)

