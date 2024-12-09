import torch
import torch.nn as nn
from torch.hub import load_state_dict_from_url
from mouse_vision.loss_functions.loss_utils import build_norm_layer

__all__ = [
    "ResNet",
    "resnet18",
    "resnet34",
    "resnet50",
    "resnet101",
    "resnet152",
    "resnet18_64x64_input",
    "resnet34_64x64_input",
    "resnet50_64x64_input",
    "resnet101_64x64_input",
    "resnet152_64x64_input",
    "resnet18_ir",
    "resnet18_simclr",
    "resnet18_simclr_nosyncbn",
    "resnext50_32x4d",
    "resnet18_simsiam",
    "resnet18_ir_64x64",
    "resnet18_ir_preavgpool_64x64",
    "resnet34_ir_64x64",
    "resnet34_ir_preavgpool_64x64",
    "resnet50_ir_64x64",
    "resnet50_ir_preavgpool_64x64",
    "resnet101_ir_64x64",
    "resnet101_ir_preavgpool_64x64",
    "resnet152_ir_64x64",
    "resnet152_ir_preavgpool_64x64",
    "resnext101_32x8d",
    "wide_resnet50_2",
    "wide_resnet101_2",
    "resnet18_relative_location",
    "resnet50_relative_location",
    "resnet18_mocov2",
    "resnet50_mocov2",
    "resnet50_barlow_twins",
]


model_urls = {
    "resnet18": "https://download.pytorch.org/models/resnet18-5c106cde.pth",
    "resnet34": "https://download.pytorch.org/models/resnet34-333f7ec4.pth",
    "resnet50": "https://download.pytorch.org/models/resnet50-19c8e357.pth",
    "resnet101": "https://download.pytorch.org/models/resnet101-5d3b4d8f.pth",
    "resnet152": "https://download.pytorch.org/models/resnet152-b121ed2d.pth",
    "resnext50_32x4d": "https://download.pytorch.org/models/resnext50_32x4d-7cdf4587.pth",
    "resnext101_32x8d": "https://download.pytorch.org/models/resnext101_32x8d-8ba56ff5.pth",
    "wide_resnet50_2": "https://download.pytorch.org/models/wide_resnet50_2-95faca4d.pth",
    "wide_resnet101_2": "https://download.pytorch.org/models/wide_resnet101_2-32ee1156.pth",
}


def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1):
    """3x3 convolution with padding"""
    return nn.Conv2d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=dilation,
        groups=groups,
        bias=False,
        dilation=dilation,
    )


def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution"""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(BasicBlock, self).__init__()
        if isinstance(norm_layer, dict):
            self.norm1_name, norm1 = build_norm_layer(norm_layer, planes, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(norm_layer, planes, postfix=2)
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.norm1_name = "bn1"
            norm1 = norm_layer(planes)
            self.norm2_name = "bn2"
            norm2 = norm_layer(planes)

        if groups != 1 or base_width != 64:
            raise ValueError("BasicBlock only supports groups=1 and base_width=64")
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu2(out)

        return out


class Bottleneck(nn.Module):
    # Bottleneck in torchvision places the stride for downsampling at 3x3 convolution(self.conv2)
    # while original implementation places the stride at the first 1x1 convolution(self.conv1)
    # according to "Deep residual learning for image recognition"https://arxiv.org/abs/1512.03385.
    # This variant is also known as ResNet V1.5 and improves accuracy according to
    # https://ngc.nvidia.com/catalog/model-scripts/nvidia:resnet_50_v1_5_for_pytorch.

    expansion = 4

    def __init__(
        self,
        inplanes,
        planes,
        stride=1,
        downsample=None,
        groups=1,
        base_width=64,
        dilation=1,
        norm_layer=None,
    ):
        super(Bottleneck, self).__init__()
        width = int(planes * (base_width / 64.0)) * groups
        if isinstance(norm_layer, dict):
            self.norm1_name, norm1 = build_norm_layer(norm_layer, width, postfix=1)
            self.norm2_name, norm2 = build_norm_layer(norm_layer, width, postfix=2)
            self.norm3_name, norm3 = build_norm_layer(
                norm_layer, planes * self.expansion, postfix=3
            )
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.norm1_name = "bn1"
            norm1 = norm_layer(width)
            self.norm2_name = "bn2"
            norm2 = norm_layer(width)
            self.norm3_name = "bn3"
            norm3 = norm_layer(planes * self.expansion)

        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width)
        self.add_module(self.norm1_name, norm1)
        self.relu1 = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(width, width, stride, groups, dilation)
        self.add_module(self.norm2_name, norm2)
        self.relu2 = nn.ReLU(inplace=True)
        self.conv3 = conv1x1(width, planes * self.expansion)
        self.add_module(self.norm3_name, norm3)
        self.relu3 = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    @property
    def norm2(self):
        return getattr(self, self.norm2_name)

    @property
    def norm3(self):
        return getattr(self, self.norm3_name)

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.norm1(out)
        out = self.relu1(out)

        out = self.conv2(out)
        out = self.norm2(out)
        out = self.relu2(out)

        out = self.conv3(out)
        out = self.norm3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu3(out)

        return out


class ResNet(nn.Module):
    def __init__(
        self,
        block,
        layers,
        num_classes=1000,
        zero_init_residual=False,
        groups=1,
        width_per_group=64,
        replace_stride_with_dilation=None,
        norm_layer=None,
        drop_final_fc=False,
        pre_avgpool=False,
    ):
        super(ResNet, self).__init__()
        self.inplanes = 64
        if isinstance(norm_layer, dict):
            self.norm1_name, norm1 = build_norm_layer(
                norm_layer, self.inplanes, postfix=1
            )
        else:
            if norm_layer is None:
                norm_layer = nn.BatchNorm2d
            self.norm1_name = "bn1"
            norm1 = norm_layer(self.inplanes)

        self._norm_layer = norm_layer

        self.dilation = 1
        if replace_stride_with_dilation is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilation = [False, False, False]
        if len(replace_stride_with_dilation) != 3:
            raise ValueError(
                "replace_stride_with_dilation should be None "
                "or a 3-element tuple, got {}".format(replace_stride_with_dilation)
            )
        self.groups = groups
        self.base_width = width_per_group
        self.conv1 = nn.Conv2d(
            3, self.inplanes, kernel_size=7, stride=2, padding=3, bias=False
        )
        self.add_module(self.norm1_name, norm1)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(
            block, 128, layers[1], stride=2, dilate=replace_stride_with_dilation[0]
        )
        self.layer3 = self._make_layer(
            block, 256, layers[2], stride=2, dilate=replace_stride_with_dilation[1]
        )
        self.layer4 = self._make_layer(
            block, 512, layers[3], stride=2, dilate=replace_stride_with_dilation[2]
        )
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        if drop_final_fc:
            if pre_avgpool:
                self.avgpool = nn.Identity()
            self.fc = nn.Identity()
        else:
            assert pre_avgpool is False
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode="fan_out", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    @property
    def norm1(self):
        return getattr(self, self.norm1_name)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            if not isinstance(norm_layer, dict):
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    norm_layer(planes * block.expansion),
                )
            else:
                downsample = nn.Sequential(
                    conv1x1(self.inplanes, planes * block.expansion, stride),
                    build_norm_layer(norm_layer, planes * block.expansion)[1],
                )
        layers = []
        layers.append(
            block(
                self.inplanes,
                planes,
                stride,
                downsample,
                self.groups,
                self.base_width,
                previous_dilation,
                norm_layer,
            )
        )
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(
                block(
                    self.inplanes,
                    planes,
                    groups=self.groups,
                    base_width=self.base_width,
                    dilation=self.dilation,
                    norm_layer=norm_layer,
                )
            )

        return nn.Sequential(*layers)

    def _forward_impl(self, x):
        # See note [TorchScript super()]
        x = self.conv1(x)
        x = self.norm1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x

    def forward(self, x):
        return self._forward_impl(x)


def _resnet(arch, block, layers, pretrained, progress, **kwargs):
    model = ResNet(block, layers, **kwargs)
    if pretrained:
        state_dict = load_state_dict_from_url(model_urls[arch], progress=progress)
        model.load_state_dict(state_dict)
    return model


def resnet18(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_64x64_input(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained with 64x64 imagenet inputs. Weights will not be loaded from checkpoint,
    which is why pretrained is always False.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet("resnet18", BasicBlock, [2, 2, 2, 2], pretrained, progress, **kwargs)


def resnet18_ir(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet18_ir_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet18_ir_preavgpool_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Same as resnet18_ir_64x64 but pre avg pool (useful for finetuning).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        pre_avgpool=True,
        **kwargs
    )


def resnet18_simsiam(pretrained=False, progress=True, tpu=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the SimSiam loss, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        # Note: when evaluating for neural fits in validation this won't matter
        # since they are the same function (just different flags) which will use F.BatchNorm
        # in eval mode, that's why we don't need a resnet18_simclr_tpu function
        norm_layer=dict(type="TPUSyncBN" if tpu else "SyncBN"),
        **kwargs
    )


def resnet18_simclr(pretrained=False, progress=True, tpu=False, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the SimCLR loss, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        # Note: when evaluating for neural fits in validation this won't matter
        # since they are the same function (just different flags) which will use F.BatchNorm
        # in eval mode, that's why we don't need a resnet18_simclr_tpu function
        norm_layer=dict(type="TPUSyncBN" if tpu else "SyncBN"),
        **kwargs
    )


def resnet18_simclr_nosyncbn(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the SimCLR loss, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.
    We also do not synchronize Batch norm statistics across cores.
    This only might matter if you are doing multi-gpu training or TPU training, but not for single gpu training..
    As SyncBN is currently unsupported on TPU, use this model as a control to test
    against performance of resnet18_simclr in the multi-gpu context.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        # we use a dict so that build_norm_layer is called, to name parameters for LARS optimizer to exclude for weight decay
        norm_layer=dict(type="BN"),
        **kwargs
    )


def resnet18_relative_location(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained on relative location task.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        # we use a dict so that build_norm_layer is called to keep in consistency with downstream head & neck
        # though it's not strictly necessary here since it is the same fct as default batch norm
        norm_layer=dict(type="BN"),
        **kwargs
    )


def resnet18_mocov2(pretrained=False, progress=True, **kwargs):
    r"""ResNet-18 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the MocoV2, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet18",
        BasicBlock,
        [2, 2, 2, 2],
        pretrained,
        progress,
        drop_final_fc=True,
        # we use a dict so that build_norm_layer is called, to name parameters for LARS optimizer to exclude for weight decay
        norm_layer=dict(type="BN"),
        **kwargs
    )


def resnet34(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet34_64x64_input(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet("resnet34", BasicBlock, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet34_ir_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet34 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet34_ir_preavgpool_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-34 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

      Same as resnet34_ir_64x64 but pre avg pool (useful for finetuning).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet34",
        BasicBlock,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        pre_avgpool=True,
        **kwargs
    )


def resnet50(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_64x64_input(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet("resnet50", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs)


def resnet50_ir_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet50_ir_preavgpool_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Same as resnet50_ir_64x64 but pre avg pool (useful for finetuning).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        pre_avgpool=True,
        **kwargs
    )


def resnet50_relative_location(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    But trained on the Relative Location task.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        # we use a dict so that build_norm_layer is called to keep in consistency with downstream head & neck
        # though it's not strictly necessary here since it is the same fct as default batch norm
        norm_layer=dict(type="BN"),
        **kwargs
    )


def resnet50_mocov2(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the MocoV2, hence we set the drop_final_fc
    argument to True as we are only using the resnet18 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        # we use a dict so that build_norm_layer is called to keep in consistency with downstream head & neck
        # though it's not strictly necessary here since it is the same fct as default batch norm
        norm_layer=dict(type="BN"),
        **kwargs
    )


def resnet101(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet101_64x64_input(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet101", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet101_ir_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet101 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet101_ir_preavgpool_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-101 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Same as resnet101_ir_64x64 but pre avg pool (useful for finetuning).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet101",
        Bottleneck,
        [3, 4, 23, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        pre_avgpool=True,
        **kwargs
    )


def resnet152(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnet152_64x64_input(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet152", Bottleneck, [3, 8, 36, 3], pretrained, progress, **kwargs
    )


def resnet152_ir_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet152 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        **kwargs
    )


def resnet152_ir_preavgpool_64x64(pretrained=False, progress=True, **kwargs):
    r"""ResNet-152 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    Same as resnet152_ir_64x64 but pre avg pool (useful for finetuning).

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet152",
        Bottleneck,
        [3, 8, 36, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        pre_avgpool=True,
        **kwargs
    )


def resnext50_32x4d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-50 32x4d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 4
    return _resnet(
        "resnext50_32x4d", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def resnext101_32x8d(pretrained=False, progress=True, **kwargs):
    r"""ResNeXt-101 32x8d model from
    `"Aggregated Residual Transformation for Deep Neural Networks" <https://arxiv.org/pdf/1611.05431.pdf>`_
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["groups"] = 32
    kwargs["width_per_group"] = 8
    return _resnet(
        "resnext101_32x8d", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def wide_resnet50_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-50-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet50_2", Bottleneck, [3, 4, 6, 3], pretrained, progress, **kwargs
    )


def wide_resnet101_2(pretrained=False, progress=True, **kwargs):
    r"""Wide ResNet-101-2 model from
    `"Wide Residual Networks" <https://arxiv.org/pdf/1605.07146.pdf>`_
    The model is the same as ResNet except for the bottleneck number of channels
    which is twice larger in every block. The number of channels in outer 1x1
    convolutions is the same, e.g. last block in ResNet-50 has 2048-512-2048
    channels, and in Wide ResNet-50-2 has 2048-1024-2048.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    kwargs["width_per_group"] = 64 * 2
    return _resnet(
        "wide_resnet101_2", Bottleneck, [3, 4, 23, 3], pretrained, progress, **kwargs
    )


def resnet50_barlow_twins(pretrained=False, progress=True, **kwargs):
    r"""ResNet-50 model from
    `"Deep Residual Learning for Image Recognition" <https://arxiv.org/pdf/1512.03385.pdf>`_

    But trained using the instance discrimination loss on 64x64 imagenet inputs, hence we set the drop_final_fc
    argument to True as we are only using the resnet50 backbone.

    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        progress (bool): If True, displays a progress bar of the download to stderr
    """
    pretrained = False
    return _resnet(
        "resnet50",
        Bottleneck,
        [3, 4, 6, 3],
        pretrained,
        progress,
        drop_final_fc=True,
        zero_init_residual=True,
        **kwargs
    )
