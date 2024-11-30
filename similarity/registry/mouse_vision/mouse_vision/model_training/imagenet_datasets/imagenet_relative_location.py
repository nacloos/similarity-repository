import torch
from PIL import Image
from torchvision.transforms import RandomCrop, Compose, ToTensor, Normalize
import torchvision.transforms.functional as TF
from mouse_vision.model_training.imagenet_datasets.imagenet_base import ImageNetBase
from mouse_vision.core.constants import IMAGENET_MEAN, IMAGENET_STD

__all__ = ["ImageNetRelativeLocation"]

def image_to_patches(image,
                     num_patches_per_side=3, # split of patches per image side
                     patch_spacing=21): # jitter of each patch from each grid
    """Crop num_patches_per_side x num_patches_per_side patches from input image.
    Args:
        image (PIL Image): input image.
    Returns:
        list[PIL Image]: A list of cropped patches.
    """
    h, w = image.size
    h_grid = h // num_patches_per_side
    w_grid = w // num_patches_per_side
    h_patch = h_grid - patch_spacing
    w_patch = w_grid - patch_spacing
    assert h_patch > 0 and w_patch > 0
    patches = []
    for i in range(num_patches_per_side):
        for j in range(num_patches_per_side):
            p = TF.crop(image, i * h_grid, j * w_grid, h_grid, w_grid)
            p = RandomCrop((h_patch, w_patch))(p)
            patches.append(p)
    return patches

class ImageNetRelativeLocation(ImageNetBase):
    """
    ImageNet data set class that also returns two "views" of the same image.

    Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/188778c62519d731eaa40b7a70105f298c15e902/openselfsup/datasets/relative_loc.py

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory fo ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self,
                 is_train,
                 imagenet_dir,
                 image_transforms,
                 center_patch_idx=4,
                 num_patches_per_side=3,
                 patch_spacing=21):
        super(ImageNetRelativeLocation, self).__init__(
            is_train, imagenet_dir, image_transforms
        )
        self.num_patches_per_side = num_patches_per_side
        self.num_patches = self.num_patches_per_side*self.num_patches_per_side
        self.patch_spacing = patch_spacing
        self.center_patch_idx = center_patch_idx
        assert(self.center_patch_idx < self.num_patches)
        self.format_transforms = Compose([ToTensor(), Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD)])

    def __getitem__(self, index):
        image_label = list(self.dataset.__getitem__(index))
        image = image_label[0]  # Assuming (C, H, W)
        assert isinstance(image, Image.Image)
        patches = image_to_patches(image,
                                   num_patches_per_side=self.num_patches_per_side,
                                   patch_spacing=self.patch_spacing)
        patches = [self.format_transforms(p) for p in patches]
        perms = []
        # create a list of patch pairs
        [perms.append(torch.cat((patches[i], patches[self.center_patch_idx]), dim=0)) for i in range(self.num_patches) if i != self.center_patch_idx]
        patch_images = torch.stack(perms)
        # create corresponding labels for patch pairs
        patch_labels = torch.LongTensor([0, 1, 2, 3, 4, 5, 6, 7])
        return tuple([patch_images, patch_labels])  # 8(2C)HW, 8

