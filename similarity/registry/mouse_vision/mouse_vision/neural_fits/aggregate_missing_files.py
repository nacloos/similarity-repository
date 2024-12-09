import os
import pickle
import numpy as np
from collections import defaultdict

from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.models.model_layers import MODEL_LAYERS
from mouse_vision.core.default_dirs import NEURAL_FIT_RESULTS_DIR_NEW
from mouse_vision.reliability.spec_map_utils import generate_spec_map

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
    "all_models": ALL_MODELS,
    "alexnet_ir_224": ["alexnet_ir_224x224"],
    "six_ir_224": ["simplified_mousenet_six_stream_visp_3x3_ir_224x224"],
    "dual_ir_224": ["simplified_mousenet_dual_stream_visp_3x3_ir_224x224"],
    "single_ir_224": ["simplified_mousenet_single_stream_ir_224x224"],
    "untrained_alexnet_64": ["untrained_alexnet_64x64_input_pool_6"],
    "intermediate_res": INTERMEDIATE_RES_MODELS,
}

def find_missing(results_dir, models, map_type="pls", check_per_specimen=False, datasets=["calcium","neuropixels"],
                 splithalf_r_thresh=None):
    incompletes = dict()
    for dataset in datasets:
        if check_per_specimen:
            if splithalf_r_thresh is not None:
                visual_area_spec_map = generate_spec_map(
                    dataset_name=dataset, equalize_source_units=False,
                    splithalf_r_thresh=splithalf_r_thresh,
                )
            else:
                visual_area_spec_map = generate_spec_map(
                    dataset_name=dataset, equalize_source_units=False
                )

        incompletes[dataset] = dict()

        for model_name in models:

            # arch_name is the based architecture name of an untrained model
            if "untrained" == model_name.split("_")[0]:
                arch_name = model_name[len("untrained") + 1 :]
            else:
                arch_name = model_name

            for visual_area in VISUAL_AREAS:
                if check_per_specimen:
                    specimens = [k for k in visual_area_spec_map[visual_area].keys() if isinstance(k, np.int64)]

                if splithalf_r_thresh is not None:
                    fname = os.path.join(
                        results_dir, f"{dataset}sphr{splithalf_r_thresh}/{model_name}/{visual_area}/{map_type}/"
                    )
                else:
                    fname = os.path.join(
                        results_dir, f"{dataset}/{model_name}/{visual_area}/{map_type}/"
                    )

                for layer in MODEL_LAYERS[arch_name]:
                    layer = layer.replace("/", "_")
                    _fname = fname + f"{layer}.pkl"

                    if os.path.isfile(_fname):
                        continue
                    else:
                        if check_per_specimen:
                            if splithalf_r_thresh is not None:
                                fname_spec = os.path.join(
                                    results_dir, f"per_specimen/{dataset}sphr{splithalf_r_thresh}/{model_name}/{visual_area}/{map_type}/{layer}/"
                                )
                            else:
                                fname_spec = os.path.join(
                                    results_dir, f"per_specimen/{dataset}/{model_name}/{visual_area}/{map_type}/{layer}/"
                                )
                            for specimen in specimens:
                                _fname_spec = fname_spec + f"{specimen}.pkl"
                                if os.path.isfile(_fname_spec):
                                    continue
                                else:
                                    if model_name not in incompletes[dataset].keys():
                                        incompletes[dataset][model_name] = defaultdict(dict)
                                    if visual_area not in incompletes[dataset][model_name].keys():
                                        incompletes[dataset][model_name][visual_area] = defaultdict(list)
                                    incompletes[dataset][model_name][visual_area][layer].append(specimen)
                        else:
                            if model_name not in incompletes[dataset].keys():
                                incompletes[dataset][model_name] = defaultdict(list)
                            incompletes[dataset][model_name][visual_area].append(layer)

    return incompletes

def print_incompletes(data):
    for d in data.keys():
        for model in data[d].keys():
            for v in data[d][model].keys():
                missing_layers = data[d][model][v]
                print(f"{d} {model} {v} {missing_layers}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--model-group", type=str, default="all_models")
    parser.add_argument("--file-save-name", type=str, default="incomplete_jobs.pkl")
    parser.add_argument("--check-per-specimen", type=bool, default=False)
    parser.add_argument("--datasets", type=str, default="calcium,neuropixels")
    parser.add_argument("--splithalf-r-thresh", type=float, default=None)

    args = parser.parse_args()

    incompletes = find_missing(NEURAL_FIT_RESULTS_DIR_NEW, MODEL_REGISTRY[args.model_group], map_type="pls", check_per_specimen=args.check_per_specimen, datasets=args.datasets.split(","),
                               splithalf_r_thresh=args.splithalf_r_thresh)
    print(incompletes)
    print_incompletes(incompletes)

    incompletes_fname = os.path.join(NEURAL_FIT_RESULTS_DIR_NEW, args.file_save_name)
    pickle.dump(incompletes, open(incompletes_fname, "wb"))

