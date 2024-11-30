import os
import pickle

import torch
import numpy as np

import torchvision.transforms as transforms

from mouse_vision.neural_data.model_data import generate_model_comparison_data
from mouse_vision.models.model_paths import MODEL_PATHS
from mouse_vision.models.model_layers import MODEL_LAYERS, MODEL_LAYER_CONCATS
from mouse_vision.models.model_transforms import MODEL_TRANSFORMS
from mouse_vision.core.constants import IMAGENET_MEAN, IMAGENET_STD
from mouse_vision.core.default_dirs import MODEL_FEATURES_SAVE_DIR
from mouse_vision.core.dataloader_utils import get_image_array_dataloader
from mouse_vision.core.model_loader_utils import load_model
from mouse_vision.core.feature_extractor import (
    FeatureExtractor,
    CustomFeatureExtractor,
    get_layer_features,
)


class ModelFeatures:
    def __init__(self, dataset, arch_name, model_family, use_64px_input=False):
        """
            use_64px_input: (bool) whether or not to use 64 px transforms for
                            neural data stimuli, even if the model was trained
                            on different resolutions.
        """
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
        self.use_64px_input = use_64px_input

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
        if not self.use_64px_input:
            img_transforms_arr = [transforms.ToPILImage()] + MODEL_TRANSFORMS[
                self.model_name
            ]["val"]
            img_transforms = transforms.Compose(img_transforms_arr)
        else:
            # If use 64 px input, then construct default 64 px transforms here
            img_transforms_arr = [
                transforms.ToPILImage(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.Resize(64),
                transforms.ToTensor(),
                transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
            ]
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
        for layer_name in MODEL_LAYERS[self.model_name]:

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


def construct_filename(model_name, dataset, use_64px_input=False):
    # Set up filename for the model features
    if use_64px_input:
        save_dir = os.path.join(MODEL_FEATURES_SAVE_DIR, f"{dataset}", "64px_features")
    else:
        save_dir = os.path.join(MODEL_FEATURES_SAVE_DIR, f"{dataset}")

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
    mf = ModelFeatures(dataset, args.arch_name, model_family, use_64px_input=args.use_64px)
    features = mf.get_model_features()

    # Save features
    fname = construct_filename(args.arch_name, dataset, use_64px_input=args.use_64px)
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
    parser.add_argument("--use-64px", type=bool, default=False)
    args = parser.parse_args()

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

    SHI_MODELS = ["shi_mousenet", "shi_mousenet_vispor5"]

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

    OTHER_MODELS = [
        "alexnet_ir_224x224",
        "simplified_mousenet_six_stream_visp_3x3_ir_224x224",
    ]

    SHORT_ALEXNET_MODELS = [
        "alexnet_two_64x64",
        "alexnet_three_64x64",
        "alexnet_four_64x64",
        "alexnet_five_64x64",
        "alexnet_six_64x64",
    ]

    OTHER_CONTRASTIVE_ALEXNET_MODELS = ["alexnet_bn_simsiam_64x64", "alexnet_bn_simclr_64x64", "alexnet_bn_mocov2_64x64"]

    SUP_ALEXNET_IR_TRANSFORM_MODELS = ["alexnet_bn_64x64_input_pool_6_with_ir_transforms", "alexnet_64x64_input_pool_6_with_ir_transforms"]

    ALL_MODELS = (
        PRIMATE_MODELS
        + UNSUPERVISED_PRIMATE_MODELS
        + SHI_MODELS
        + SINGLE_STREAM_MODELS
        + DUAL_STREAM_MODELS
        + SIX_STREAM_MODELS
        + OTHER_MODELS
        + SHORT_ALEXNET_MODELS
        + OTHER_CONTRASTIVE_ALEXNET_MODELS
        + SUP_ALEXNET_IR_TRANSFORM_MODELS
    )

    MODEL_REGISTRY = {
        "primate_models": PRIMATE_MODELS,
        "unsupervised_primate_models": UNSUPERVISED_PRIMATE_MODELS,
        "short_alexnet_models": SHORT_ALEXNET_MODELS,
        "shi_models": SHI_MODELS,
        "single_stream_models": SINGLE_STREAM_MODELS,
        "dual_stream_models": DUAL_STREAM_MODELS,
        "six_stream_model": SIX_STREAM_MODELS,
        "other_models": OTHER_MODELS,
        "all_models": ALL_MODELS,
        "alexnet_truncated_rl": ["alexnet_64x64_rl_scratch_truncated"],
        "alexnet_ir_dmlocomotion": ["alexnet_ir_dmlocomotion"],
        "other_conunsup_alexnet": OTHER_CONTRASTIVE_ALEXNET_MODELS,
        "sup_alexnet_ir_64": SUP_ALEXNET_IR_TRANSFORM_MODELS,
    }

    if args.model_group is None:
        assert args.model_name is not None
        models = [args.model_name]
    else:
        models = MODEL_REGISTRY[args.model_group]

    print(f"Getting features for {models}.")
    main(args, models)


