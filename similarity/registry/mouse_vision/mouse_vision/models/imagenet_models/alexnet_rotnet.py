import copy

import torch
import torch.nn as nn

__all__ = ["AlexNetRotNet", "alexnet_rotnet", "alexnet_rotnet_transfer"]


class AlexNetRotNet(nn.Module):
    """
    Note that we have a different file for this AlexNet version so that we can
    use the same model architecture as the one used in the original RotNet
    publication.
    """

    def __init__(self, pool_size=6, finetune_readout=False):
        super(AlexNetRotNet, self).__init__()
        self.pool_size = pool_size
        self.finetune_readout = finetune_readout

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.BatchNorm2d(192),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # pool5
        )

        self.readout_features = self.features[:-1]

        self.classifier = nn.Sequential(
            nn.Linear(256 * pool_size * pool_size, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
            nn.Linear(4096, 4096, bias=False),
            nn.BatchNorm1d(4096),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        if not self.finetune_readout:
            x = self.features(x)
            x = torch.flatten(x, 1)
            x = self.classifier(x)
        else:
            x = self.readout_features(x)
        return x


def alexnet_rotnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    This will be trained on rotation prediction.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetRotNet(**kwargs)
    return model


def alexnet_rotnet_transfer(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    This model will be loaded with weights trained on rotation prediction, but
    only readout features will be used for transfer on e.g., ImageNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetRotNet(finetune_readout=True, **kwargs)
    return model
