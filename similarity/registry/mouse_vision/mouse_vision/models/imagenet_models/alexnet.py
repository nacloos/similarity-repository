import torch
import torch.nn as nn

from collections import OrderedDict

from torch.hub import load_state_dict_from_url

__all__ = [
    "AlexNet",
    "AlexNetBN",
    "alexnet",
    "alexnet_bn",
    "alexnet_64x64_input_pool_6",
    "alexnet_64x64_input_pool_1",
    "alexnet_64x64_input_pool_6_with_ir_transforms",
    "alexnet_bn_64x64_input_pool_6_with_ir_transforms",
    "alexnet_relative_location",
    "alexnet_ir_64x64_input_pool_6",
    "alexnet_bn_ir_64x64_input_pool_6",
    "alexnet_ir_224x224",
    "alexnet_ir_84x84",
    "alexnet_ir_104x104",
    "alexnet_ir_124x124",
    "alexnet_ir_144x144",
    "alexnet_ir_164x164",
    "alexnet_ir_184x184",
    "alexnet_ir_204x204",
    "alexnet_ir_dmlocomotion",
    "alexnet_bn_mocov2_64x64",
    "alexnet_bn_simclr_64x64",
    "alexnet_bn_simsiam_64x64",
    "alexnet_64x64_rl_scratch_truncated",
    "alexnet_bn_barlow_twins_64x64",
    "alexnet_bn_barlow_twins",
    "alexnet_bn_vicreg_64x64",
]

model_urls = {
    "alexnet": "https://download.pytorch.org/models/alexnet-owt-4df8aa71.pth",
}


class AlexNet(nn.Module):
    def __init__(
        self, num_classes=1000, pool_size=6, drop_final_fc=False, fc6_out=False
    ):
        super(AlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        if drop_final_fc:
            assert fc6_out is False
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * pool_size * pool_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
            )
        elif fc6_out:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * pool_size * pool_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * pool_size * pool_size, 4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


class AlexNetBN(nn.Module):
    def __init__(self, num_classes=1000, pool_size=6, drop_final_fc=False):
        super(AlexNetBN, self).__init__()
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
            nn.MaxPool2d(kernel_size=3, stride=2),
        )
        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))
        if drop_final_fc:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * pool_size * pool_size, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
            )
        else:
            self.classifier = nn.Sequential(
                nn.Dropout(),
                nn.Linear(256 * pool_size * pool_size, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Dropout(),
                nn.Linear(4096, 4096),
                nn.BatchNorm1d(4096),
                nn.ReLU(inplace=True),
                nn.Linear(4096, num_classes),
            )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(**kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls["alexnet"], progress=progress)
        model.load_state_dict(state_dict)
    return model


def alexnet_64x64_rl_scratch_truncated(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained end-to-end on the rodent maze task. A truncated (four-layer) version of
    the regular AlexNet.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    # First initialize full AlexNet
    model = AlexNet(pool_size=6, **kwargs)

    # Next, truncate up to fourth convolutional layer (including ReLU)
    model = torch.nn.Sequential(OrderedDict([("features", model.features[:10])]))

    return model


def alexnet_64x64_input_pool_6_with_ir_transforms(
    pretrained=False, progress=True, **kwargs
):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 imagenet inputs. Avg pooling size of 6. Also trained using
    the transforms of the instance recognition model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pool_size=6, **kwargs)
    return model


def alexnet_64x64_input_pool_6(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 imagenet inputs. Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pool_size=6, **kwargs)
    return model


def alexnet_64x64_input_pool_1(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 imagenet inputs. Avg pooling size of 1.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pool_size=1, **kwargs)
    return model


def alexnet_relative_location(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Avg pooling size of 6. Returns output at fc6 layer, according to
    Fig. 3 of https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Doersch_Unsupervised_Visual_Representation_ICCV_2015_paper.pdf

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pool_size=6, fc6_out=True, **kwargs)
    return model


def alexnet_ir_64x64_input_pool_6(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained using the instance discrimination loss with 64x64 imagenet inputs. Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNet(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But with batch norm.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(**kwargs)
    return model


def alexnet_bn_simsiam_64x64(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the SimSiam loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_simclr_64x64(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the SimCLR loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_mocov2_64x64(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the MoCov2 loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_64x64_input_pool_6_with_ir_transforms(
    pretrained=False, progress=True, **kwargs
):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 imagenet inputs. Avg pooling size of 6. Also trained using
    the transforms of the instance recognition model.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, **kwargs)
    return model


def alexnet_bn_ir_64x64_input_pool_6(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the instance discrimination loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_84x84(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 84x84 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_104x104(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 104x104 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_124x124(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 124x124 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_144x144(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 144x144 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_164x164(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 164x164 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_184x184(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 184x184 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_204x204(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 204x204 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_224x224(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 224x224 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_ir_dmlocomotion(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    Trained using the instance discrimination loss with 64x64 dmlocomotion inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_barlow_twins_64x64(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the BarlowTwins loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_barlow_twins(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the Barlow Twins loss.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model


def alexnet_bn_vicreg_64x64(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.
    Same as above but with BN.
    trained using the VICReg loss with 64x64 imagenet inputs.
    Avg pooling size of 6.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    model = AlexNetBN(pool_size=6, drop_final_fc=True, **kwargs)
    return model
