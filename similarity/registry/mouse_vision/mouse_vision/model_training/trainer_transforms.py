"""
Contains information on image transformations used by different trainers
(e.g. supervised, self-supervised, etc) during training and validation.
"""
import numpy as np
import random
from torchvision import transforms
from mouse_vision.core.constants import CIFAR10_MEAN, CIFAR10_STD
from mouse_vision.core.constants import IMAGENET_MEAN, IMAGENET_STD
from PIL import Image, ImageFilter, ImageOps


class TwinsTransform:
    def __init__(self, tr1, tr2):
        self.transform = transforms.Compose(tr1)
        self.transform_prime = transforms.Compose(tr2)

    def __call__(self, x):
        y1 = self.transform(x)
        y2 = self.transform_prime(x)
        return y1, y2


class GaussianBlur(object):
    """Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/3272f765c5b7d5bee9772233a5a4d7e3fb66e5bf/openselfsup/datasets/pipelines/transforms.py#L83-L97
    """

    def __init__(self, sigma_min, sigma_max):
        self.sigma_min = sigma_min
        self.sigma_max = sigma_max

    def __call__(self, img):
        sigma = np.random.uniform(self.sigma_min, self.sigma_max)
        img = img.filter(ImageFilter.GaussianBlur(radius=sigma))
        return img

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class RandomAppliedTrans(object):
    """Randomly applied transformations.
    Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/3272f765c5b7d5bee9772233a5a4d7e3fb66e5bf/openselfsup/datasets/pipelines/transforms.py#L22-L39
    Args:
        transforms_arr (list): List of transformations.
        p (float): Probability.
    """

    def __init__(self, transforms_arr, p=0.5):
        self.trans = transforms.RandomApply(transforms_arr, p=p)

    def __call__(self, img):
        return self.trans(img)

    def __repr__(self):
        repr_str = self.__class__.__name__
        return repr_str


class GaussianBlurBarlowTwins(object):
    """
    Taken from: https://github.com/facebookresearch/barlowtwins/blob/a655214c76c97d0150277b85d16e69328ea52fd9/main.py#L267-L276
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            sigma = random.random() * 1.9 + 0.1
            return img.filter(ImageFilter.GaussianBlur(sigma))
        else:
            return img


class Solarization(object):
    """
    Taken from: https://github.com/facebookresearch/barlowtwins/blob/a655214c76c97d0150277b85d16e69328ea52fd9/main.py#L279-L287
    """

    def __init__(self, p):
        self.p = p

    def __call__(self, img):
        if random.random() < self.p:
            return ImageOps.solarize(img)
        else:
            return img


TRAINER_TRANSFORMS = dict()

TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer"] = dict()
TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer"]["train"] = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
]
TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer"]["val"] = [
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
]

TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer_64x64"] = dict()
TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer_64x64"]["train"] = [
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
]
TRAINER_TRANSFORMS["SupervisedCIFAR10Trainer_64x64"]["val"] = [
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=CIFAR10_MEAN, std=CIFAR10_STD),
]

TRAINER_TRANSFORMS["SupervisedImageNetTrainer"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# val from: https://pytorch.org/hub/pytorch_vision_inception_v3/
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_inception"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_inception"]["train"] = [
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_inception"]["val"] = [
    transforms.Resize(299),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# val from: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/xception.py#L22
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_xception"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_xception"]["train"] = [
    transforms.RandomResizedCrop(299),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_xception"]["val"] = [
    transforms.Resize(333),
    transforms.CenterCrop(299),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]

# val from: https://github.com/Cadene/pretrained-models.pytorch/blob/master/pretrainedmodels/models/nasnet_mobile.py#L29
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_nasnetamobile"] = dict()
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_nasnetamobile"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]
TRAINER_TRANSFORMS["SupervisedImageNetTrainer_nasnetamobile"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
]

# Instance discrimination transforms
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_32x32"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_32x32"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_32x32"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(32),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_44x44"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_44x44"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_44x44"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(44),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_84x84"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_84x84"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_84x84"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(84),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_104x104"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_104x104"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(104),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_104x104"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(104),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_124x124"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_124x124"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(124),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_124x124"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(124),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_144x144"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_144x144"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(144),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_144x144"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(144),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_164x164"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_164x164"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(164),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_164x164"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(164),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_184x184"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_184x184"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(184),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_184x184"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(184),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_204x204"] = dict()
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_204x204"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(204),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["InstanceDiscriminationTrainer_204x204"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(204),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["DMLocomotionInstanceDiscriminationTrainer"] = dict()
TRAINER_TRANSFORMS["DMLocomotionInstanceDiscriminationTrainer"]["train"] = [
    transforms.RandomResizedCrop(
        64, scale=(0.4, 1.0)
    ),  # increasing the scale since input images already small (64x64)
    transforms.RandomGrayscale(p=0.2),
    transforms.ColorJitter(0.4, 0.4, 0.4, 0.4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["DMLocomotionInstanceDiscriminationTrainer"]["val"] = [
    transforms.Resize(256),  # we keep this since we val on neural data
    transforms.CenterCrop(224),  # we keep this since we val on neural data
    transforms.Resize(64),  # we keep this since we val on neural data
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# SimCLR transforms
# from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/configs/selfsup/simclr/r50_bs256_ep200.py#L29-L53
TRAINER_TRANSFORMS["SimCLRTrainer"] = dict()
TRAINER_TRANSFORMS["SimCLRTrainer"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SimCLRTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["SimCLRTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["SimCLRTrainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.8, contrast=0.8, saturation=0.8, hue=0.2
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SimCLRTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# RotNet transforms
TRAINER_TRANSFORMS["RotNetTrainer"] = dict()
TRAINER_TRANSFORMS["RotNetTrainer"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["RotNetTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["RotNetTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["RotNetTrainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(0.5),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["RotNetTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# Relative Location transforms
# from: https://github.com/open-mmlab/OpenSelfSup/blob/aa62006c6e0fb3ee9474dbe8e009b65af35e8e06/configs/selfsup/relative_loc/r50.py#L33-L45
TRAINER_TRANSFORMS["RelativeLocationTrainer"] = dict()
TRAINER_TRANSFORMS["RelativeLocationTrainer"]["train"] = [
    transforms.Resize(292),
    transforms.RandomCrop(255),
    transforms.RandomGrayscale(p=0.66),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["RelativeLocationTrainer"]["val"] = [
    transforms.Resize(292),
    transforms.RandomCrop(255),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# MOCOv2 transforms
# from: https://github.com/open-mmlab/OpenSelfSup/blob/796cd68297d57233ad59aaf0cf0d7f75926bfa6f/configs/selfsup/moco/r50_v2.py#L30-L59
TRAINER_TRANSFORMS["MoCov2Trainer"] = dict()
TRAINER_TRANSFORMS["MoCov2Trainer"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["MoCov2Trainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["MoCov2Trainer_64x64"] = dict()
TRAINER_TRANSFORMS["MoCov2Trainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.RandomHorizontalFlip(),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["MoCov2Trainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# SimSiam transforms
TRAINER_TRANSFORMS["SimSiamTrainer"] = dict()
TRAINER_TRANSFORMS["SimSiamTrainer"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SimSiamTrainer"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

TRAINER_TRANSFORMS["SimSiamTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["SimSiamTrainer_64x64"]["train"] = [
    transforms.RandomResizedCrop(224, scale=(0.2, 1.0)),
    transforms.RandomHorizontalFlip(),
    RandomAppliedTrans(
        transforms_arr=[
            transforms.ColorJitter(
                brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1
            )
        ],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    RandomAppliedTrans(
        transforms_arr=[GaussianBlur(sigma_min=0.1, sigma_max=2.0)], p=0.5
    ),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["SimSiamTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# Depth Prediction Transforms
TRAINER_TRANSFORMS["DepthPredictionTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["DepthPredictionTrainer_64x64"]["train"] = [
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]
TRAINER_TRANSFORMS["DepthPredictionTrainer_64x64"]["val"] = [
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.Resize(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

# BarlowTwins Transforms
# from: https://github.com/facebookresearch/barlowtwins/blob/a655214c76c97d0150277b85d16e69328ea52fd9/main.py#L292-L321
BT_postfix = [
    transforms.ToTensor(),
    transforms.Normalize(mean=IMAGENET_MEAN, std=IMAGENET_STD),
]

BT_TR1_prefix = [
    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlurBarlowTwins(p=1.0),
    Solarization(p=0.0),
]

BT_TR2_prefix = [
    transforms.RandomResizedCrop(224, interpolation=Image.BICUBIC),
    transforms.RandomHorizontalFlip(p=0.5),
    transforms.RandomApply(
        [transforms.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.2, hue=0.1)],
        p=0.8,
    ),
    transforms.RandomGrayscale(p=0.2),
    GaussianBlurBarlowTwins(p=0.1),
    Solarization(p=0.2),
]

BT_TR1 = BT_TR1_prefix + BT_postfix
BT_TR1_64x64 = BT_TR1_prefix + [transforms.Resize(64)] + BT_postfix

BT_TR2 = BT_TR2_prefix + BT_postfix
BT_TR2_64x64 = BT_TR2_prefix + [transforms.Resize(64)] + BT_postfix

# transforms during linear training & evaluation, from: https://github.com/facebookresearch/barlowtwins/blob/main/evaluate.py#L125-L137
BT_TR_prefix = [transforms.RandomResizedCrop(224), transforms.RandomHorizontalFlip()]
BT_VAL_prefix = [transforms.Resize(256), transforms.CenterCrop(224)]

BT_TR = BT_TR_prefix + BT_postfix
BT_TR_64x64 = BT_TR_prefix + [transforms.Resize(64)] + BT_postfix

BT_VAL = BT_VAL_prefix + BT_postfix
BT_VAL_64x64 = BT_VAL_prefix + [transforms.Resize(64)] + BT_postfix

TRAINER_TRANSFORMS["BarlowTwinsTrainer"] = dict()
TRAINER_TRANSFORMS["BarlowTwinsTrainer"]["tr1"] = BT_TR1
TRAINER_TRANSFORMS["BarlowTwinsTrainer"]["tr2"] = BT_TR2
TRAINER_TRANSFORMS["BarlowTwinsTrainer"]["train"] = BT_TR
TRAINER_TRANSFORMS["BarlowTwinsTrainer"]["val"] = BT_VAL

TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"]["tr1"] = BT_TR1_64x64
TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"]["tr2"] = BT_TR2_64x64
TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"]["train"] = BT_TR_64x64
TRAINER_TRANSFORMS["BarlowTwinsTrainer_64x64"]["val"] = BT_VAL_64x64

# from: https://github.com/facebookresearch/vicreg/blob/cc1ef4f1584a26770308a9b8f1538e490a6d4b43/augmentations.py
TRAINER_TRANSFORMS["VICRegTrainer"] = dict()
TRAINER_TRANSFORMS["VICRegTrainer"]["tr1"] = BT_TR1
TRAINER_TRANSFORMS["VICRegTrainer"]["tr2"] = BT_TR2
TRAINER_TRANSFORMS["VICRegTrainer"]["train"] = BT_TR
TRAINER_TRANSFORMS["VICRegTrainer"]["val"] = BT_VAL

TRAINER_TRANSFORMS["VICRegTrainer_64x64"] = dict()
TRAINER_TRANSFORMS["VICRegTrainer_64x64"]["tr1"] = BT_TR1_64x64
TRAINER_TRANSFORMS["VICRegTrainer_64x64"]["tr2"] = BT_TR2_64x64
TRAINER_TRANSFORMS["VICRegTrainer_64x64"]["train"] = BT_TR_64x64
TRAINER_TRANSFORMS["VICRegTrainer_64x64"]["val"] = BT_VAL_64x64
