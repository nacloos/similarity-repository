import torch
import torch.nn as nn
from mouse_vision.models.imagenet_models.alexnet import (
    alexnet_64x64_input_pool_6,
    alexnet_64x64_input_pool_1,
)
from mouse_vision.models.imagenet_models.resnet import resnet18_64x64_input
from mouse_vision.models.imagenet_models.vgg import vgg16_64x64_input
from mouse_vision.models.imagenet_models.shi_mousenet import (
    shi_mousenet,
    shi_mousenet_vispor5,
)
from mouse_vision.models.imagenet_models.parallel_stream_mousenet import (
    simplified_mousenet_six_stream,
    simplified_mousenet_six_stream_visp_3x3,
    simplified_mousenet_six_stream_vispor_only,
    simplified_mousenet_six_stream_vispor_only_visp_3x3,
    simplified_mousenet_dual_stream,
    simplified_mousenet_dual_stream_visp_3x3,
    simplified_mousenet_dual_stream_vispor_only,
    simplified_mousenet_dual_stream_vispor_only_visp_3x3,
    simplified_mousenet_single_stream_base,
)


__all__ = [
    "alexnet_64x64_input_pool_6_cifar10",
    "alexnet_64x64_input_pool_1_cifar10",
    "resnet18_64x64_input_cifar10",
    "vgg16_64x64_input_cifar10",
    "shi_mousenet_cifar10",
    "shi_mousenet_vispor5_cifar10",
    "simplified_mousenet_six_stream_cifar10",
    "simplified_mousenet_six_stream_visp_3x3_cifar10",
    "simplified_mousenet_six_stream_vispor_only_cifar10",
    "simplified_mousenet_six_stream_vispor_only_visp_3x3_cifar10",
    "simplified_mousenet_dual_stream_cifar10",
    "simplified_mousenet_dual_stream_visp_3x3_cifar10",
    "simplified_mousenet_dual_stream_vispor_only_cifar10",
    "simplified_mousenet_dual_stream_vispor_only_visp_3x3_cifar10",
    "simplified_mousenet_single_stream_cifar10",
    "simplified_mousenet_dual_stream_visp_3x3_bn_cifar10",
    "simplified_mousenet_six_stream_visp_3x3_bn_cifar10",
]


def alexnet_64x64_input_pool_6_cifar10(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 image inputs. Avg pooling size of 6.
    Weights will be loaded from checkpoint.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return alexnet_64x64_input_pool_6(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def alexnet_64x64_input_pool_1_cifar10(pretrained=False, progress=True, **kwargs):
    r"""AlexNet model architecture from the
    `"One weird trick..." <https://arxiv.org/abs/1404.5997>`_ paper.

    But trained with 64x64 image  inputs. Avg pooling size of 1.
    Weights will be loaded from checkpoint.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return alexnet_64x64_input_pool_1(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def resnet18_64x64_input_cifar10(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained with 64x64 image inputs. Weights will be loaded from checkpoint,
    which is why pretrained is always False.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return resnet18_64x64_input(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def vgg16_64x64_input_cifar10(pretrained=False, progress=True, **kwargs):
    r"""VGG 16-layer model (configuration "D")
    `"Very Deep Convolutional Networks For Large-Scale Image Recognition" <https://arxiv.org/pdf/1409.1556.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return vgg16_64x64_input(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def shi_mousenet_cifar10(pretrained=False, progress=True, **kwargs):
    r"""Shi et al. 2020 MouseNet model architecture from their
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf> paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output areas (str):
    """
    return shi_mousenet(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def shi_mousenet_vispor5_cifar10(pretrained=False, progress=True, **kwargs):
    r"""Variant of Shi et al. 2020 MouseNet model architecture
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf>
    that trains classifier on VISpor5 output only and with no avg pooling.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    return shi_mousenet_vispor5(
        num_classes=10, pretrained=pretrained, progress=progress, **kwargs
    )


def simplified_mousenet_six_stream_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier.
    """
    return simplified_mousenet_six_stream(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    return simplified_mousenet_six_stream_visp_3x3(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_six_stream_visp_3x3_bn_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam. Has batch normalization (BN).
    """
    return simplified_mousenet_six_stream_visp_3x3(
        num_classes=10, norm_layer=dict(type="BN"), pretrained=pretrained, **kwargs
    )


def simplified_mousenet_six_stream_vispor_only_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are taken only from VISpor prior to the fully-
    connected classifier.
    """
    return simplified_mousenet_six_stream_vispor_only(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_six_stream_vispor_only_visp_3x3_cifar10(
    pretrained=False, **kwargs
):
    """
    Simplified MouseNet architecture with six parallel streams, one stream
    for each of the following visual areas: "VISl", "VISrl", "VISal", "VISli",
    "VISpm", "VISpl". Outputs are taken only from VISpor prior to the fully-
    connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    return simplified_mousenet_six_stream_vispor_only_visp_3x3(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_dual_stream_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are concatenated from VISpor and
    VISam prior to the fully-connected classifier.
    """
    return simplified_mousenet_dual_stream(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    return simplified_mousenet_dual_stream_visp_3x3(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_dual_stream_visp_3x3_bn_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are concatenated from VISpor and VISam prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam. Has batch normalization (BN).
    """
    return simplified_mousenet_dual_stream_visp_3x3(
        num_classes=10, norm_layer=dict(type="BN"), pretrained=pretrained, **kwargs
    )


def simplified_mousenet_dual_stream_vispor_only_cifar10(pretrained=False, **kwargs):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are taken only from VISpor prior to
    the fully-connected classifier.
    """
    return simplified_mousenet_dual_stream_vispor_only(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_dual_stream_vispor_only_visp_3x3_cifar10(
    pretrained=False, **kwargs
):
    """
    Simplified MouseNet architecture with two parallel streams: the apparently
    "dorsal" and "ventral" streams. Outputs are taken only from VISpor prior to
    the fully-connected classifier. Outputs from VISp are processed with a
    3x3 convolution (stride 2) before VISpor / VISam.
    """
    return simplified_mousenet_dual_stream_vispor_only_visp_3x3(
        num_classes=10, pretrained=pretrained, **kwargs
    )


def simplified_mousenet_single_stream_cifar10(pretrained=False, **kwargs):

    return simplified_mousenet_single_stream_base(
        num_classes=10, pretrained=pretrained, norm_layer=dict(type="BN"), **kwargs
    )
