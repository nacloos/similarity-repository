import os
import pickle

import torch
import numpy as np

import torchvision.transforms as transforms

from mouse_vision.neural_data.model_data import generate_model_comparison_data
from mouse_vision.models.model_paths import MODEL_PATHS
from mouse_vision.models.model_layers import MODEL_LAYERS_EXTENDED
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.default_dirs import MODEL_FEATURES_EXTENDED_SAVE_DIR
from mouse_vision.core.dataloader_utils import get_image_array_dataloader
from mouse_vision.core.model_loader_utils import load_model
from mouse_vision.core.feature_extractor import (
    FeatureExtractor,
    CustomFeatureExtractor,
    get_layer_features,
)


class ModelFeatures:
    def __init__(self, dataset, arch_name, model_family):
        self.dataset = dataset

        # Untrained models will have the following arch name syntax:
        # "untrained_{arch_name}"
        if "untrained" == arch_name.split("_")[0]:
            trained = False
            model_name = arch_name[len("untrained") + 1 :]
        else:
            trained = True
            model_name = arch_name

        # Grab custom model path if applicable. Note that if the model_name is
        # "untrained_{arch_name}" it won't exist in MODEL_PATHS and therefore,
        # model_path will remain None, which is the desired behaviour (we don't
        # want to load weights for untrained models anyway).
        model_path = None
        if arch_name in MODEL_PATHS.keys():
            model_path = MODEL_PATHS[arch_name]

        # Load model
        try:
            print("Trying 'state_dict' key to load parameters")
            model, _ = load_model(
                model_name,
                trained=trained,
                model_path=model_path,
                model_family=model_family,
                state_dict_key="state_dict",
            )
        except:
            print("'state_dict' key failed, using 'model_state_dict'")
            model, _ = load_model(
                model_name,
                trained=trained,
                model_path=model_path,
                model_family=model_family,
                state_dict_key="model_state_dict",
            )

        self.model = model
        self.model_name = model_name
        self.arch_name = arch_name

        self.stim_dataloader = self.get_stim_dataloader()
        self.feature_extractor = self.get_feature_extractor()

    def get_stim_dataloader(self):

        # We only need the stimuli, so some argument values are arbitrary
        _, stimuli = generate_model_comparison_data(
            center_trials=True,
            return_stimuli=True,
            dataset_name=self.dataset,
            separate_by_animal=False,
        )

        assert stimuli.ndim == 4
        assert stimuli.shape[3] == 3  # RGB channels
        assert (
            self.model_name in MODEL_TRANSFORMS.keys()
        ), f"{self.model_name} does not exist."

        # Use the val transforms of the architecture/model, but prepend with
        # ToPILImage() since neural images are tensors instead of jpegs
        img_transforms_arr = [transforms.ToPILImage()] + MODEL_TRANSFORMS[
            self.model_name
        ]["val"]
        img_transforms = transforms.Compose(img_transforms_arr)

        # Use some arbitrary labels
        n_stim = stimuli.shape[0]
        labels = np.arange(n_stim).astype(int)

        # Get stimuli dataloader
        stim_dataloader = get_image_array_dataloader(
            image_array=stimuli,
            labels=labels,
            img_transform=img_transforms,
            batch_size=256,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
        )

        return stim_dataloader

    def get_feature_extractor(self):
        # Dictionary based models
        if (
            "mousenet" in self.model_name
            or "alexnet_64x64_input_dict" in self.model_name
        ):
            feature_extractor = CustomFeatureExtractor(
                dataloader=self.stim_dataloader, vectorize=True, debug=False
            )
        else:
            feature_extractor = FeatureExtractor(
                dataloader=self.stim_dataloader, vectorize=True, debug=False
            )

        return feature_extractor

    def get_model_features(self):
        # Get features and store it as a dictionary with the following organization:
        # {layer_name0: layer_features0, layer_name1: layer_features1, ...}

        layer_features = dict()
        for layer_name in MODEL_LAYERS_EXTENDED[self.model_name]:

            # Get model activations
            curr_layer_features = get_layer_features(
                feature_extractor=self.feature_extractor,
                layer_name=layer_name,
                model=self.model,
                model_name=self.model_name,
            )

            print(
                f"Current layer: {layer_name}, Activations of shape: {curr_layer_features.shape}"
            )
            layer_features[layer_name] = curr_layer_features

        return layer_features


def seed_random(seed):
    # Set random seed for untrained model initializations
    torch.manual_seed(seed)
    np.random.seed(seed)


def construct_filename(model_name, dataset):
    # Set up filename for the model features
    save_dir = os.path.join(MODEL_FEATURES_EXTENDED_SAVE_DIR, f"{dataset}")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    fname = os.path.join(save_dir, f"{model_name}.pkl")
    return fname


def load_and_save_features_for_dataset(args, dataset):
    assert dataset in ["calcium", "neuropixels"]

    # Set model family
    if "cifar10" in args.arch_name:
        model_family = "cifar10"
    else:
        model_family = "imagenet"

    # Load features
    mf = ModelFeatures(dataset, args.arch_name, model_family)
    features = mf.get_model_features()

    # Save features
    fname = construct_filename(args.arch_name, dataset)
    pickle.dump(features, open(fname, "wb"))


def load_and_save_features_for_model(args):
    seed_random(int(args.seed))
    load_and_save_features_for_dataset(args, "calcium")

    seed_random(int(args.seed))
    load_and_save_features_for_dataset(args, "neuropixels")


def main(args, models):
    for model_name in models:
        args.arch_name = model_name

        print(f"Saving features for {args.arch_name}...")
        load_and_save_features_for_model(args)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--model-group", type=str, default=None)
    parser.add_argument("--model-name", type=str, default=None)
    args = parser.parse_args()

    ALL_MODELS = [
        "vgg16_64x64_input",
        "resnet18_64x64_input",
        "resnet34_64x64_input",
        "resnet50_64x64_input",
        "resnet101_64x64_input",
        "resnet152_64x64_input",
        "vgg16_ir_64x64",
        "resnet18_ir_64x64",
        "resnet34_ir_64x64",
        "resnet50_ir_64x64",
        "resnet101_ir_64x64",
        "resnet152_ir_64x64",
    ]

    MODEL_REGISTRY = {
        "all_models": ALL_MODELS,
    }

    if args.model_group is None:
        assert args.model_name is not None
        models = [args.model_name]
    else:
        models = MODEL_REGISTRY[args.model_group]

    print(f"Getting features for {models}.")
    main(args, models)


