import numpy as np

import torch
import torchvision

from torch.utils import data
from torchvision import transforms, datasets

from mouse_vision.core.constants import IMAGENET_MEAN, IMAGENET_STD


class ArrayDataset(data.Dataset):
    """
    General dataset constructor using an array of images and labels.

    Arguments:
        image_array : numpy array of shape (N, H, W, 3)
        labels      : numpy array of shape (N,)
        t           : torchvision.transforms instance
    """

    def __init__(self, image_array, labels, t=None):
        assert image_array.shape[0] == labels.shape[0]
        assert t is not None

        self.transforms = t
        self.image_array = image_array
        self.labels = labels
        self.n_images = image_array.shape[0]

    def __getitem__(self, index):
        inputs = self.transforms(self.image_array[index, :, :, :])
        labels = self.labels[index]
        return inputs, labels

    def __len__(self):
        return self.n_images


def _acquire_data_loader(dataset, batch_size, shuffle, num_workers, pin_memory=True):
    assert isinstance(dataset, data.Dataset)
    loader = data.DataLoader(
        dataset,
        batch_size,
        shuffle=shuffle,
        pin_memory=pin_memory,
        num_workers=num_workers,
    )
    return loader


def get_image_array_dataloader(
    image_array,
    labels,
    img_transform,
    batch_size=256,
    shuffle=False,
    num_workers=8,
    pin_memory=True,
):
    """
    Inputs: 
        image_array   : (numpy.ndarray) (N, H, W, 3)
        labels        : (numpy.ndarray) (N,)
        img_transform : (torchvision.transforms) instance for image transformations

    Outputs:
        dataloader  : (torch.utils.data.DataLoader) for the image array
    """

    assert image_array.shape[0] == labels.shape[0]

    dataset = ArrayDataset(image_array, labels, t=img_transform)
    dataloader = _acquire_data_loader(
        dataset, batch_size, shuffle, num_workers, pin_memory=pin_memory
    )
    return dataloader


def duplicate_channels(gray_images):
    """
    Converts single channel grayscale images into rgb channel images

    Input:
        gray_images : (N,H,W)

    Output:
        rgb : (N,H,W,3)
    """
    n, dim0, dim1 = gray_images.shape[:3]
    rgb = np.empty((n, dim0, dim1, 3), dtype=np.uint8)
    rgb[:, :, :, 2] = rgb[:, :, :, 1] = rgb[:, :, :, 0] = gray_images
    return rgb
