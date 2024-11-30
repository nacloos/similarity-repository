import os
import pickle

import torch
import numpy as np

from mouse_vision.core.constants import VISUAL_AREAS
from mouse_vision.core.default_dirs import NEURAL_FIT_RESULTS_DIR
from mouse_vision.models.model_paths import MODEL_PATHS
from mouse_vision.models.model_layers import MODEL_LAYERS, MODEL_LAYER_CONCATS
from mouse_vision.neural_fits.neural_predictions import NeuralPredictions
from mouse_vision.neural_mappers.map_param_utils import generate_map_param_grid

def seed_neural_fits(seed):
    torch.manual_seed(seed)
    np.random.seed(seed)

def filename(base_save_nm,
             num_splits=10,
             train_frac=0.5,
             center_trials=True,
             separate_by_animal=False,
             mode="spearman_brown_split_half",
             concat_scheme=None,
             new_format=True):

    # e.g. VISp2/3
    file_nm = base_save_nm.replace("/", "_")
    if concat_scheme is not None:
        file_nm += f"_concatscheme{concat_scheme}"
    if new_format:
        file_nm += f"_num_splits{num_splits}"
        file_nm += f"_train_frac{train_frac}"
        file_nm += f"_center_trials{center_trials}"
        file_nm += f"_per_animal{separate_by_animal}"
        file_nm += f"_model{mode}"
    file_nm += ".pkl"
    return file_nm

def main(args):
    seed_neural_fits(int(args.seed))

    # Untrained models will have the following arch name syntax: "untrained_{arch_name}"
    if "untrained" == args.arch_name.split('_')[0]:
        trained = False
        arch_name = args.arch_name[len("untrained")+1:]
    else:
        trained = True
        arch_name = args.arch_name

    if args.map_type == "pls":
        cv_map_param_grid = generate_map_param_grid(
            map_type=args.map_type,
            pls_n_components=[10,20,30,40,50,100],
            pls_scale=False
        )
    elif args.map_type == "identity":
        cv_map_param_grid = generate_map_param_grid(map_type=args.map_type)
    else:
        raise ValueError(f"{args.map_type} not supported yet.")

    # Grab custom model path if applicable
    model_path = None
    if args.arch_name in MODEL_PATHS.keys():
        model_path = MODEL_PATHS[args.arch_name]

    center_trials = True if not args.bootstrap_trials else False
    print(f"Doing fits for {arch_name}, Trained: {trained}, Center Trials: {center_trials}, Per Animal: {args.separate_by_animal}")
    neural_predictions = NeuralPredictions(
        arch_name=arch_name,
        visual_area=args.visual_area,
        dataset_name=args.dataset,
        map_type=args.map_type,
        map_cv_params=cv_map_param_grid,
        map_n_cv_splits=5,
        num_splits=args.num_splits,
        train_frac=args.train_frac,
        trained=trained,
        model_params_path=model_path,
        model_family=args.model_family,
        state_dict_key=args.state_dict_key,
        center_trials=center_trials,
        separate_by_animal=args.separate_by_animal,
        mode=args.mode
    )

    # Using args.arch_name because we want to differentiate between results from
    # untrained models and results from trained models (i.e., untrained models will
    # have results directory: .../untrained_{arch_name}/{visual_area}/{map_type}/{layer_name}.pkl)
    results_dir = os.path.join(NEURAL_FIT_RESULTS_DIR, f"{args.dataset}/{args.arch_name}/{args.visual_area}/{args.map_type}")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)

    if args.concat_scheme is not None:
        concat_scheme = MODEL_LAYER_CONCATS[arch_name][args.concat_scheme]
        for group_name, group_layers in concat_scheme.items():
            print(f"Fitting to concatenated layers: {group_layers} within group: {group_name}")
            scores, best_params = neural_predictions.fit(group_layers)
            group_save_name = filename(base_save_nm=group_name,
                                         num_splits=args.num_splits,
                                         train_frac=args.train_frac,
                                         center_trials=center_trials,
                                         separate_by_animal=args.separate_by_animal,
                                         mode=args.mode,
                                         concat_scheme=args.concat_scheme)
            results_fname = os.path.join(results_dir, group_save_name)
            pickle.dump({"scores": scores, "best_params": best_params}, open(results_fname, "wb"))
    else:
        for layer_name in MODEL_LAYERS[arch_name]:
            print(f"Fitting for layer: {layer_name}")
            scores, best_params = neural_predictions.fit(layer_name)
            layer_save_name = filename(base_save_nm=layer_name,
                                         num_splits=args.num_splits,
                                         train_frac=args.train_frac,
                                         center_trials=center_trials,
                                         separate_by_animal=args.separate_by_animal,
                                         mode=args.mode,
                                         concat_scheme=args.concat_scheme)
            results_fname = os.path.join(results_dir, layer_save_name)
            pickle.dump({"scores": scores, "best_params": best_params}, open(results_fname, "wb"))

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--arch-name", type=str, default="alexnet")
    parser.add_argument("--model-family", type=str, default=None)
    parser.add_argument("--state-dict-key", type=str, default="state_dict")
    parser.add_argument("--concat-scheme", type=str, default=None)
    parser.add_argument("--visual-area", type=str, default="VISp", choices=VISUAL_AREAS)
    parser.add_argument("--dataset", type=str, default="neuropixels", choices=["neuropixels", "calcium"])
    parser.add_argument("--map-type", type=str, default="pls", choices=["pls", "corr", "factored", "identity"])
    parser.add_argument("--num-splits", type=int, default=10)
    parser.add_argument("--train-frac", type=float, default=0.5)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--bootstrap-trials", type=bool, default=False)
    parser.add_argument("--separate-by-animal", type=bool, default=False)
    parser.add_argument("--mode", type=str, default="spearman_brown_split_half")
    args = parser.parse_args()

    main(args)

