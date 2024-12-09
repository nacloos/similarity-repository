"""
Functions to name paths, initialise models, initialise datasets
"""
import os

import torch
import torch.nn as nn
#from torchvision import models, transforms, datasets
import pathlib

# Vision paths
DATASETS = ["tiny_imagenet", "imagenet"]
ARCHITECTURES = [
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "inceptionv1",
]

# TODO: change get_filepath and get_string_filepath to be like get_embedding_folderpath so that we can use the same function in score_pair.py
def get_filepath(dataset, architecture, seed, step, layer, folder=False):
    """
    Get filepath for embedding of interest (in order to check whether it has already
    been computed
    """
    if folder:
        return os.path.join(
            EMBEDDING_PATH, dataset, architecture, str(seed), str(step), str(layer)
        )
    else:
        return os.path.join(
            EMBEDDING_PATH,
            dataset,
            architecture,
            str(seed),
            str(step),
            str(layer),
            "rep.npy",
        )


def get_string_filepath(dataset, architecture, seed, step, layer):
    return "{head}/{dataset}/{architecture}/{seed}/{step}/{layer}".format(
        head=EMBEDDING_PATH,
        dataset=dataset,
        architecture=architecture,
        seed=seed,
        step=step,
        layer=layer,
    )


def get_embedding_folderpath(
    dataset: str,
    architecture: str,
    seed: int,
    step: int,
) -> pathlib.Path:
    """
    Return path of folder containing embedding arrays corresponding to:
    - layers of model specified by architecture, seed and step
    - inputs from dataset

    Args:
        dataset (str): name of the dataset on which to compute embedding, eg "tiny_imagenet"
        architecture (str): name of model architecture, eg "resnet18"
        seed (int): seed used to train model
        step (int): number of training steps to train model

    Returns:
        pathlib.Path: path to embedding folder
    """
    path_suffix = f"embeddings/{dataset}/{architecture}/{seed}/{step}/"
    return SCRATCH_PATH / pathlib.Path(path_suffix)


def get_checkpoint_filepath(architecture: str, seed: int, step: int) -> pathlib.Path:
    """
    Return path to model checkpoint specified by architecture, seed and step

    Args:
        architecture (str): name of model architecture, eg "resnet18"
        seed (int): seed used to train model
        step (int): number of training steps to train model

    Returns:
        pathlib.Path: path to model checkpoint
    """
    path_suffix = f"checkpoints/{architecture}/seed_{seed}_step_{step}.pt"
    return DATA_PATH / pathlib.Path(path_suffix)


def initialise_model(architecture: str) -> nn.Module:
    """
    Return initialised network of a given architecture
    Currently: only works for resnet18, resnet34, resnet50, resnet101, resnet152

    Args:
        architecture (str): name of model architecture, eg "resnet18"

    Returns:
        nn.Module: initialised network
    """
    assert architecture in ARCHITECTURES
    assert architecture != "inceptionv1"

    if architecture == "resnet18":
        blocked_model = models.resnet18(pretrained=True)
    if architecture == "resnet34":
        blocked_model = models.resnet34(pretrained=True)
    if architecture == "resnet50":
        blocked_model = models.resnet50(pretrained=True)
    if architecture == "resnet101":
        blocked_model = models.resnet101(pretrained=True)
    if architecture == "resnet152":
        blocked_model = models.resnet152(pretrained=True)
    return blocked_model


def initialise_dataset(
    dataset: str, sample_size: int, sample_seed: int, normalize=True
):
    """
    Return Dataset object corresponding to a dataset name

    Args:
        dataset (str): name of dataset
        sample_size (int): number of inputs to subsample
        sample_seed (int): seed to use when subsampling inputs

    Returns:
        torch.utils.data.Dataset: Dataset object corresponding to the name
    """
    dataset_folderpath = DATA_PATH / pathlib.Path("datasets/")
    if dataset == "tiny_imagenet":
        ds = datasets.ImageFolder(
            root=dataset_folderpath / pathlib.Path("tiny-imagenet-200/val/"),
            transform=transforms.ToTensor(),
        )
    if dataset == "imagenet":
        if normalize:
            normalize = transforms.Normalize(
                mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
            )
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                    normalize,
                ]
            )
        else:
            transform = transforms.Compose(
                [
                    transforms.Resize(256),
                    transforms.CenterCrop(224),
                    transforms.ToTensor(),
                ]
            )
        ds = datasets.ImageFolder(
            root=dataset_folderpath / pathlib.Path("imagenet/val/"),
            transform=transform,
        )
    if sample_seed == None:
        torch.manual_seed(0)
    else:
        torch.manual_seed(sample_seed)
    random_indices = torch.randperm(len(ds))[:sample_size]
    ds = torch.utils.data.Subset(ds, indices=random_indices)
    return ds, random_indices
