import h5py
import copy

import numpy as np

from mouse_vision.sklearn_transfer_datasets import BaseDataset
from mouse_vision.core.default_dirs import HVM_DATA_PATH
from mouse_vision.core.dataloader_utils import (
    get_image_array_dataloader,
    duplicate_channels,
)


__all__ = ["HvmDataset"]


class HvmDataset(BaseDataset):
    """
    Class for HVM dataset stimuli and stimuli meta data. Also contains function
    for obtaining a PyTorch dataloader for the stimuli.

    Arguments:
        image_transforms : (torchvision.transforms) image transforms
        datapath         : (string) path to the HVM dataset
    """
    def __init__(self, image_transforms, datapath=HVM_DATA_PATH, name="hvm_dataset"):
        self.image_transforms = image_transforms

        data = h5py.File(datapath, "r")
        self.stim, self.stim_meta = self._extract_image_data(data)
        self.dataloader = None

        self.name = name

    def _extract_image_data(self, data):
        images = np.array(data["images"])

        # Image metadata
        all_meta = dict()
        for meta in data["image_meta"].keys():
            all_meta[meta] = np.array(data["image_meta"][meta])

        # b'Animals', b'Boats', b'Cars', b'Chairs', b'Faces', b'Fruits',
        # b'Planes', b'Tables'
        labels = copy.deepcopy(all_meta["category"])
        labels[labels == b"Animals"] = 1
        labels[labels == b"Boats"] = 2
        labels[labels == b"Cars"] = 3
        labels[labels == b"Chairs"] = 4
        labels[labels == b"Faces"] = 5
        labels[labels == b"Fruits"] = 6
        labels[labels == b"Planes"] = 7
        labels[labels == b"Tables"] = 8
        labels = labels.astype(np.int)
        all_meta["category_index"] = labels

        # Change instance labels into integers
        instance_labels = copy.deepcopy(all_meta["object_name"])
        assert len(np.unique(instance_labels)) == 64
        for i, instance_label in enumerate(np.unique(instance_labels)):
            assert (instance_labels == instance_label).sum() == 90
            instance_labels[instance_labels == instance_label] = i
        all_meta["instance_index"] = instance_labels.astype(int)

        # Duplicate channels since original images are grayscale
        images = duplicate_channels(images)

        return images, all_meta

    def get_stim_metadata(self):
        return self.stim_meta

    def get_dataloader(self):
        if self.dataloader is None:
            self.dataloader = get_image_array_dataloader(
                self.stim,
                self.stim_meta["category_index"],
                self.image_transforms,
                batch_size=256,
                shuffle=False,
                num_workers=8,
                pin_memory=True,
            )

        return self.dataloader


if __name__ == "__main__":
    import torch.nn as nn
    hvm = HvmDataset(nn.Identity())
    dloader = hvm.get_dataloader()
    for i, (x, y) in enumerate(dloader):
        print(f"Batch {i+1}:", x.shape, y.shape)

    stim_meta = hvm.get_stim_metadata()
    print(stim_meta.keys())
    print(stim_meta["category_index"].shape)

