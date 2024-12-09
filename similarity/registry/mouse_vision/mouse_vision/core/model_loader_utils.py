import os
import socket
from mouse_vision.core.default_dirs import TORCH_HOME
from collections import OrderedDict

os.environ["TORCH_HOME"] = TORCH_HOME

import torch

import mouse_vision.models.imagenet_models as imagenet_models
import mouse_vision.models.cifar_models as cifar_models
from mouse_vision.models.model_layers import MODEL_LAYERS


def get_model(arch_name, trained, model_family=None, **kwargs):
    """
    Inputs:
        arch_name : (string) Name of deep net architecture.
        trained   : (boolean) Whether or not to load a pretrained model.

    Outputs:
        model     : (torch.nn.DataParallel) model
    """
    if model_family is None:
        if "cifar10" in arch_name:
            model_family = "cifar10"
        else:
            model_family = "imagenet"
    try:
        print(
            f"Loading {arch_name}. Pretrained: {trained}. Model Family: {model_family}."
        )
        if model_family == "imagenet":
            model_family = imagenet_models
        elif model_family == "cifar10":
            model_family = cifar_models
        else:
            raise ValueError
        model = model_family.__dict__[arch_name](pretrained=trained, **kwargs)
    except:
        raise ValueError(f"{arch_name} not implemented yet.")

    return model


def load_model(
    arch_name,
    trained=False,
    model_path=None,
    model_family="imagenet",
    state_dict_key="state_dict",
    **kwargs,
):
    """
    Inputs:
        arch_name  : (string) Name of architecture (e.g. "resnet18")
        trained    : (boolean) Whether to load a pretrained or trained model.
        model_path : (string) Path of model checkpoint from which to load
                     weights.

    Outputs:
        model      : (torch.nn.DataParallel) model
    """
    model = get_model(arch_name, trained=trained, model_family=model_family, **kwargs)
    # Load weights if params file is given.
    if model_path is not None and trained:
        try:
            params = torch.load(model_path, map_location="cpu")
        except:
            raise ValueError(f"Could not open file: {model_path}")

        assert (
            state_dict_key in params.keys()
        ), f"{state_dict_key} not in params dictionary."
        sd = params[state_dict_key]
        # adapted from: https://discuss.pytorch.org/t/solved-keyerror-unexpected-key-module-encoder-embedding-weight-in-state-dict/1686/13
        new_sd = OrderedDict()
        for k, v in sd.items():
            if k.startswith("module."):
                name = k[7:]  # remove 'module.' of dataparallel/DDP
            else:
                name = k
            new_sd[name] = v
        model.load_state_dict(new_sd)
        print(f"Loaded parameters from {model_path}")

    # Set model to eval mode
    model.eval()

    assert arch_name in MODEL_LAYERS.keys(), f"Layers for {arch_name} not identified."
    layers = MODEL_LAYERS[arch_name]

    return model, layers
