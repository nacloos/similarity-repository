import os

from torch.utils import data
from torchvision import datasets

__all__ = ["ImageNetBase"]


class ImageNetBase(data.Dataset):
    """
    Base class for obtaining ImageNet data set.

    Arguments:
        is_train         : (boolean) if training or validation set
        imagenet_dir     : (string) base directory for ImageNet images.
        image_transforms : (torchvision.Transforms) object for image transforms. For
                           example: transforms.Compose([transforms.ToTensor()])
    """

    def __init__(self, is_train, imagenet_dir, image_transforms):
        # Assumes imagenet_dir organization is:
        # /PATH/TO/IMAGENET/{train, val}/{synsets}/*.JPEG

        super(ImageNetBase, self).__init__()
        suffix = "train" if is_train else "val"
        self.dataset = datasets.ImageFolder(
            os.path.join(imagenet_dir, suffix), transform=image_transforms
        )

    def __getitem__(self, index):
        raise NotImplementedError

    def __len__(self):
        return len(self.dataset)

