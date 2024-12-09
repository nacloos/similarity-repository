import os

import torch
import numpy as np

from PIL import Image

import torchvision.transforms.functional as trf
from torch.utils import data
from torchvision import datasets, transforms

__all__ = ["PBRNetDepth"]


def _npz_loader(path):
    sample = np.load(path)
    image = sample["image"]
    depth = sample["depth"]
    return image, depth

def _png_loader(path):
    # Do nothing since we will manually open the file
    return

def get_depth_map_transforms(image_transforms):
    """
    Helper function to modify the image transforms needed for depth map
    """

    depth_map_transforms = list()
    for tr in image_transforms.transforms:

        # For Resize transforms, change interpolation mode to "NEAREST"
        if isinstance(tr, transforms.Resize):
            new_transform = transforms.Resize(tr.size, interpolation=Image.NEAREST)

        # Get rid of normalization transformation
        elif isinstance(tr, transforms.Normalize):
            continue

        # Include the rest
        else:
            new_transform = tr

        depth_map_transforms.append(new_transform)

    return depth_map_transforms


def depth_map_normalize(depth):
    """
    Normalizes the depth map so that it's mean and std across pixels
    is 0 and 1 respectively.

    Input:
        depth : (torch.Tensor) 1xHxW

    Output:
        depth : (torch.Tensor) 1xHxW
    """
    assert depth.ndim == 3
    assert depth.shape[0] == 1
    N = depth.shape[0] * depth.shape[1] * depth.shape[2]

    mean = torch.mean(depth)
    # To avoid division by zero (see tf.image.per_image_standardization)
    std = max(torch.std(depth), 1.0 / np.sqrt(N))

    depth = (depth - mean) / std
    return depth


def crop(image, depth):
    """
    In this random crop, we are assuming the output shape is 224x224xC.

    Inputs:
        image : (numpy.ndarray) HxWx3
        depth : (numpy.ndarray) HxWx1

    Outputs:
        image : (numpy.ndarray) 224x224x3
        depth : (numpy.ndarray) 224x224x1
    """
    assert image.shape[:2] == depth.shape[:2]
    height = image.shape[0]
    width = image.shape[1]

    random_height = np.random.randint(low=0, high=height-224)
    random_width = np.random.randint(low=0, high=width-224)

    image = image[random_height:random_height+224, random_width:random_width+224, :]
    depth = depth[random_height:random_height+224, random_width:random_width+224, :]

    assert image.shape[:2] == depth.shape[:2]
    assert image.shape[0] == image.shape[1] == 224
    return image, depth


def flip(image, depth):
    # Randomly flips horizontally the image and depth map, which are both
    # numpy arrays.
    prob_flip = np.random.rand()
    if prob_flip < 0.5:
        image = np.fliplr(image)
        depth = np.fliplr(depth)

    return image, depth


class PBRNetDepth(data.Dataset):
    def __init__(self, is_train, pbrnet_dir, image_transforms):
        super(PBRNetDepth, self).__init__()

        # pbrnet_dir images: /PATH/TO/PBRNET/{train,val}/data/*.png
        # pbrnet_dir depth maps: /PATH/TO/PBRNET/{train_depth,val_depth}/data/*.png
        if is_train:
            self.train = True
            data_dir = os.path.join(pbrnet_dir, "train")
            self.depth_data_dir = os.path.join(pbrnet_dir, "train_depth", "data")
        else:
            self.train = False
            data_dir = os.path.join(pbrnet_dir, "val")
            self.depth_data_dir = os.path.join(pbrnet_dir, "val_depth", "data")

        self.to_pil_image = transforms.ToPILImage()
        self.image_transforms = image_transforms
        self.dataset = datasets.DatasetFolder(
            root=data_dir, loader=_png_loader, extensions=".png"
        )
        self.samples = self.dataset.samples # list of tuples (path_to_sample, class)

        depth_map_transforms = get_depth_map_transforms(self.image_transforms)
        self.depth_map_transforms = transforms.Compose(depth_map_transforms)

        print(f"Image transforms: {self.image_transforms}")
        print(f"Depth transforms: {self.depth_map_transforms}")

    def __getitem__(self, index):
        # The zero index gets the path to the sample and one index is the class index
        image_path = self.samples[index][0]
        depth_path = os.path.join(self.depth_data_dir, image_path.split('/')[-1])

        image = np.array(Image.open(image_path).convert("RGB"))
        depth_map = Image.open(depth_path)
        assert depth_map.mode == "L"
        depth_map = np.expand_dims(np.array(depth_map), axis=-1)

        # If training mode, we manually do the cropping. We also manually do
        # the flipping. In validation mode, we use the default transforms 
        # i.e., Resize(256), CenterCrop(224), ToTensor(), Resize(64), Normalize()).
        if self.train:
            image, depth_map = crop(image, depth_map)
            image, depth_map = flip(image, depth_map)

        # Going into the torchvision transforms, the image size is 224x224
        image = self.image_transforms(self.to_pil_image(image))

        # Going into the torchvision transforms, the depth map size is 224x224
        depth_map = self.depth_map_transforms(self.to_pil_image(depth_map))
        depth_map = depth_map_normalize(depth_map)

        return image, depth_map

    def __len__(self):
        return len(self.dataset)


if __name__ == "__main__":
    from mouse_vision.core.default_dirs import PBRNET_DATA_DIR

    img_trans = transforms.Compose([
        transforms.Resize(64),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0,0,0],std=[1,1,1])
    ])

    pbr_dset = PBRNetDepth(True, PBRNET_DATA_DIR, img_trans)
    print(f"Size of train set: {len(pbr_dset)}")

    pbr_dset = PBRNetDepth(False, PBRNET_DATA_DIR, img_trans)
    print(f"Size of val set: {len(pbr_dset)}")


