import torch
import torch.nn as nn
import numpy as np

__all__ = [
    "ShiMouseNet",
    "shi_mousenet",
    "shi_mousenet_vispor5",
    "shi_mousenet_vispor5_pool_4",
    "shi_mousenet_ir",
    "shi_mousenet_vispor5_ir",
]

Conv_Params = {
    "input_LGNv": {
        "in_channels": 3,
        "out_channels": 5,
        "kernel_size": 9,
        "gsh": 1,
        "gsw": 4,
        "stride": 1,
        "padding": 4,
    },
    "LGNv_VISp4": {
        "in_channels": 5,
        "out_channels": 22,
        "kernel_size": 3,
        "gsh": 1,
        "gsw": 1,
        "stride": 1,
        "padding": 1,
    },
    "VISp4_VISp2/3": {
        "in_channels": 22,
        "out_channels": 45,
        "kernel_size": 9,
        "gsh": 0.1618247181204468,
        "gsw": 4.373461393573715,
        "stride": 1,
        "padding": 4,
    },
    "VISp4_VISl4": {
        "in_channels": 22,
        "out_channels": 9,
        "kernel_size": 19,
        "gsh": 0.034912195639630085,
        "gsw": 9.829922101651555,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp4_VISrl4": {
        "in_channels": 22,
        "out_channels": 7,
        "kernel_size": 19,
        "gsh": 0.037577178073911185,
        "gsw": 9.62647371280456,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp4_VISli4": {
        "in_channels": 22,
        "out_channels": 4,
        "kernel_size": 19,
        "gsh": 0.037449852426863633,
        "gsw": 9.312342560678996,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp4_VISpl4": {
        "in_channels": 22,
        "out_channels": 8,
        "kernel_size": 19,
        "gsh": 0.037737549420137274,
        "gsw": 9.122277123908836,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp4_VISal4": {
        "in_channels": 22,
        "out_channels": 22,
        "kernel_size": 17,
        "gsh": 0.04582299564985576,
        "gsw": 8.698111152169682,
        "stride": 2,
        "padding": (7, 8, 7, 8),
    },
    "VISp4_VISpor4": {
        "in_channels": 22,
        "out_channels": 14,
        "kernel_size": 17,
        "gsh": 0.015624046737049343,
        "gsw": 8.68917420147244,
        "stride": 2,
        "padding": (7, 8, 7, 8),
    },
    "VISp2/3_VISp5": {
        "in_channels": 45,
        "out_channels": 40,
        "kernel_size": 3,
        "gsh": 0.16733836379893013,
        "gsw": 1.875687381575286,
        "stride": 1,
        "padding": 1,
    },
    "VISp2/3_VISl4": {
        "in_channels": 45,
        "out_channels": 9,
        "kernel_size": 19,
        "gsh": 0.0357372249704897,
        "gsw": 9.648758580007351,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp2/3_VISrl4": {
        "in_channels": 45,
        "out_channels": 7,
        "kernel_size": 21,
        "gsh": 0.02971095178107225,
        "gsw": 10.051600895829274,
        "stride": 2,
        "padding": (9, 10, 9, 10),
    },
    "VISp2/3_VISli4": {
        "in_channels": 45,
        "out_channels": 4,
        "kernel_size": 17,
        "gsh": 0.042097917460472346,
        "gsw": 8.982344127733013,
        "stride": 2,
        "padding": (7, 8, 7, 8),
    },
    "VISp2/3_VISpl4": {
        "in_channels": 45,
        "out_channels": 8,
        "kernel_size": 17,
        "gsh": 0.044107426612921594,
        "gsw": 8.417906307424047,
        "stride": 2,
        "padding": (7, 8, 7, 8),
    },
    "VISp2/3_VISal4": {
        "in_channels": 45,
        "out_channels": 22,
        "kernel_size": 15,
        "gsh": 0.05825007824673881,
        "gsw": 7.881177496505771,
        "stride": 2,
        "padding": (6, 7, 6, 7),
    },
    "VISp2/3_VISpor4": {
        "in_channels": 45,
        "out_channels": 14,
        "kernel_size": 19,
        "gsh": 0.011840275898830133,
        "gsw": 9.10446893778518,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISl4_VISl2/3": {
        "in_channels": 9,
        "out_channels": 22,
        "kernel_size": 9,
        "gsh": 0.1618247181204468,
        "gsw": 4.908689853497773,
        "stride": 1,
        "padding": 4,
    },
    "VISl4_VISpor4": {
        "in_channels": 9,
        "out_channels": 14,
        "kernel_size": 15,
        "gsh": 0.02228886519414421,
        "gsw": 7.210773738657816,
        "stride": 1,
        "padding": 7,
    },
    "VISrl4_VISrl2/3": {
        "in_channels": 7,
        "out_channels": 17,
        "kernel_size": 11,
        "gsh": 0.1618247181204468,
        "gsw": 5.757328262821668,
        "stride": 1,
        "padding": 5,
    },
    "VISrl4_VISpor4": {
        "in_channels": 7,
        "out_channels": 14,
        "kernel_size": 9,
        "gsh": 0.10329409855623324,
        "gsw": 4.017319565911084,
        "stride": 1,
        "padding": 4,
    },
    "VISli4_VISli2/3": {
        "in_channels": 4,
        "out_channels": 10,
        "kernel_size": 17,
        "gsh": 0.1618247181204468,
        "gsw": 8.166830772572673,
        "stride": 1,
        "padding": 8,
    },
    "VISli4_VISpor4": {
        "in_channels": 4,
        "out_channels": 14,
        "kernel_size": 19,
        "gsh": 0.018471083439349053,
        "gsw": 9.09279040974996,
        "stride": 1,
        "padding": 9,
    },
    "VISpl4_VISpl2/3": {
        "in_channels": 8,
        "out_channels": 18,
        "kernel_size": 17,
        "gsh": 0.1618247181204468,
        "gsw": 8.58362181881933,
        "stride": 1,
        "padding": 8,
    },
    "VISpl4_VISpor4": {
        "in_channels": 8,
        "out_channels": 14,
        "kernel_size": 3,
        "gsh": 0.07510132356117454,
        "gsw": 1.3667550904906651,
        "stride": 1,
        "padding": 1,
    },
    "VISal4_VISal2/3": {
        "in_channels": 22,
        "out_channels": 52,
        "kernel_size": 13,
        "gsh": 0.1618247181204468,
        "gsw": 6.43880724752149,
        "stride": 1,
        "padding": 6,
    },
    "VISal4_VISpor4": {
        "in_channels": 22,
        "out_channels": 14,
        "kernel_size": 3,
        "gsh": 0.22673891001537122,
        "gsw": 1.7552967865784397,
        "stride": 1,
        "padding": 1,
    },
    "VISpor4_VISpor2/3": {
        "in_channels": 14,
        "out_channels": 34,
        "kernel_size": 13,
        "gsh": 0.1618247181204468,
        "gsw": 6.233913519810125,
        "stride": 1,
        "padding": 6,
    },
    "VISp5_VISl4": {
        "in_channels": 40,
        "out_channels": 9,
        "kernel_size": 19,
        "gsh": 0.03388534685980559,
        "gsw": 9.751817168318247,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp5_VISrl4": {
        "in_channels": 40,
        "out_channels": 7,
        "kernel_size": 19,
        "gsh": 0.033449050398161526,
        "gsw": 9.74783302631345,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp5_VISli4": {
        "in_channels": 40,
        "out_channels": 4,
        "kernel_size": 19,
        "gsh": 0.03439205629544149,
        "gsw": 9.838764315797347,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISp5_VISpl4": {
        "in_channels": 40,
        "out_channels": 8,
        "kernel_size": 17,
        "gsh": 0.04282976789181918,
        "gsw": 8.709977858622878,
        "stride": 2,
        "padding": (7, 8, 7, 8),
    },
    "VISp5_VISal4": {
        "in_channels": 40,
        "out_channels": 22,
        "kernel_size": 15,
        "gsh": 0.05152652598864332,
        "gsw": 7.482449550640896,
        "stride": 2,
        "padding": (6, 7, 6, 7),
    },
    "VISp5_VISpor4": {
        "in_channels": 40,
        "out_channels": 14,
        "kernel_size": 19,
        "gsh": 0.012734222793549202,
        "gsw": 9.33667743711479,
        "stride": 2,
        "padding": (8, 9, 8, 9),
    },
    "VISl2/3_VISl5": {
        "in_channels": 22,
        "out_channels": 18,
        "kernel_size": 5,
        "gsh": 0.16733836379893013,
        "gsw": 2.144073754616868,
        "stride": 1,
        "padding": 2,
    },
    "VISl2/3_VISpor4": {
        "in_channels": 22,
        "out_channels": 14,
        "kernel_size": 15,
        "gsh": 0.015374443484884934,
        "gsw": 7.128501966356189,
        "stride": 1,
        "padding": 7,
    },
    "VISrl2/3_VISrl5": {
        "in_channels": 17,
        "out_channels": 14,
        "kernel_size": 5,
        "gsh": 0.16733836379893013,
        "gsw": 2.43908311658361,
        "stride": 1,
        "padding": 2,
    },
    "VISrl2/3_VISpor4": {
        "in_channels": 17,
        "out_channels": 14,
        "kernel_size": 9,
        "gsh": 0.08197499618177426,
        "gsw": 4.0596254132643175,
        "stride": 1,
        "padding": 4,
    },
    "VISli2/3_VISli5": {
        "in_channels": 10,
        "out_channels": 8,
        "kernel_size": 7,
        "gsh": 0.16733836379893013,
        "gsw": 3.1801753087496003,
        "stride": 1,
        "padding": 3,
    },
    "VISli2/3_VISpor4": {
        "in_channels": 10,
        "out_channels": 14,
        "kernel_size": 17,
        "gsh": 0.019500176808407967,
        "gsw": 8.500519928431299,
        "stride": 1,
        "padding": 8,
    },
    "VISpl2/3_VISpl5": {
        "in_channels": 18,
        "out_channels": 18,
        "kernel_size": 5,
        "gsh": 0.16733836379893013,
        "gsw": 2.5469288681099593,
        "stride": 1,
        "padding": 2,
    },
    "VISpl2/3_VISpor4": {
        "in_channels": 18,
        "out_channels": 14,
        "kernel_size": 5,
        "gsh": 0.0502379863688434,
        "gsw": 2.3936095353633857,
        "stride": 1,
        "padding": 2,
    },
    "VISal2/3_VISal5": {
        "in_channels": 52,
        "out_channels": 53,
        "kernel_size": 5,
        "gsh": 0.16733836379893013,
        "gsw": 2.756878958780832,
        "stride": 1,
        "padding": 2,
    },
    "VISal2/3_VISpor4": {
        "in_channels": 52,
        "out_channels": 14,
        "kernel_size": 1,
        "gsh": 1.2581895617439574,
        "gsw": 0.6014396216421058,
        "stride": 1,
        "padding": 0,
    },
    "VISpor2/3_VISpor5": {
        "in_channels": 34,
        "out_channels": 28,
        "kernel_size": 3,
        "gsh": 0.16733836379893013,
        "gsw": 1.724692214155501,
        "stride": 1,
        "padding": 1,
    },
    "VISl5_VISpor4": {
        "in_channels": 18,
        "out_channels": 14,
        "kernel_size": 15,
        "gsh": 0.020882959846672787,
        "gsw": 7.549486682662879,
        "stride": 1,
        "padding": 7,
    },
    "VISrl5_VISpor4": {
        "in_channels": 14,
        "out_channels": 14,
        "kernel_size": 9,
        "gsh": 0.08328209629620217,
        "gsw": 4.306473892530217,
        "stride": 1,
        "padding": 4,
    },
    "VISli5_VISpor4": {
        "in_channels": 8,
        "out_channels": 14,
        "kernel_size": 15,
        "gsh": 0.038070707825689616,
        "gsw": 7.820253760755574,
        "stride": 1,
        "padding": 7,
    },
    "VISpl5_VISpor4": {
        "in_channels": 18,
        "out_channels": 14,
        "kernel_size": 5,
        "gsh": 0.08320538639455688,
        "gsw": 2.1374090891846036,
        "stride": 1,
        "padding": 2,
    },
    "VISal5_VISpor4": {
        "in_channels": 53,
        "out_channels": 14,
        "kernel_size": 1,
        "gsh": 1.1923705757318432,
        "gsw": 0.6768369853389479,
        "stride": 1,
        "padding": 0,
    },
}

BN_Params = {
    "LGNv": {"num_features": 5},
    "VISp4": {"num_features": 22},
    "VISp2/3": {"num_features": 45},
    "VISl4": {"num_features": 9},
    "VISrl4": {"num_features": 7},
    "VISli4": {"num_features": 4},
    "VISpl4": {"num_features": 8},
    "VISal4": {"num_features": 22},
    "VISpor4": {"num_features": 14},
    "VISp5": {"num_features": 40},
    "VISl2/3": {"num_features": 22},
    "VISrl2/3": {"num_features": 17},
    "VISli2/3": {"num_features": 10},
    "VISpl2/3": {"num_features": 18},
    "VISal2/3": {"num_features": 52},
    "VISpor2/3": {"num_features": 34},
    "VISl5": {"num_features": 18},
    "VISrl5": {"num_features": 14},
    "VISli5": {"num_features": 8},
    "VISpl5": {"num_features": 18},
    "VISal5": {"num_features": 53},
    "VISpor5": {"num_features": 28},
}


class Conv2dMask(nn.Conv2d):
    """
    Conv2d with Gaussian mask
    """

    def __init__(
        self, in_channels, out_channels, kernel_size, gsh, gsw, stride=1, padding=0
    ):
        super(Conv2dMask, self).__init__(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=kernel_size,
            stride=stride,
        )
        self.mypadding = nn.ConstantPad2d(padding=padding, value=0)
        self.mask = nn.Parameter(
            torch.Tensor(
                self.make_gaussian_kernel_mask_vary_channel(
                    peak=gsh,
                    sigma=gsw,
                    kernel_size=kernel_size,
                    out_channels=out_channels,
                    in_channels=in_channels,
                )
            ),
            requires_grad=False,
        )

    def forward(self, input):
        return super(Conv2dMask, self)._conv_forward(
            self.mypadding(input), self.weight * self.mask, bias=self.bias
        )

    def make_gaussian_kernel_mask(self, peak, sigma, edge_z=1):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param edge_z: Z-score (# standard deviations) of edge of kernel
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        width = int(sigma * edge_z)
        x = np.arange(-width, width + 1)
        X, Y = np.meshgrid(x, x)
        radius = np.sqrt(X ** 2 + Y ** 2)

        probability = peak * np.exp(-(radius ** 2) / 2 / sigma ** 2)

        re = np.random.rand(len(x), len(x)) < probability
        return re

    def make_gaussian_kernel_mask_vary_channel(
        self, peak, sigma, kernel_size, out_channels, in_channels
    ):
        """
        :param peak: peak probability of non-zero weight (at kernel center)
        :param sigma: standard deviation of Gaussian probability (kernel pixels)
        :param kernel_size: kernel size of the conv2d
        :param out_channels: number of output channels of the conv2d
        :param in_channels: number of input channels of the con2d
        :return: mask in shape of kernel with True wherever kernel entry is non-zero
        """
        re = np.zeros((out_channels, in_channels, kernel_size, kernel_size))
        for i in range(out_channels):
            for j in range(in_channels):
                re[i, j, :] = self.make_gaussian_kernel_mask(peak=peak, sigma=sigma)
        return re


class ShiMouseNet(nn.Module):
    """
    torch model constructed by parameters provided in network.
    """

    def __init__(
        self,
        num_classes=1000,
        pool_size=4,
        output_areas=[
            "VISp5",
            "VISl5",
            "VISrl5",
            "VISli5",
            "VISpl5",
            "VISal5",
            "VISpor5",
        ],
        drop_final_fc=False,
    ):

        super(ShiMouseNet, self).__init__()
        self.layers = None
        self.pool_size = pool_size
        self.Convs = nn.ModuleDict()
        self.BNs = nn.ModuleDict()
        self._areas = [
            "input",
            "LGNv",
            "VISp4",
            "VISp2/3",
            "VISp5",
            "VISal4",
            "VISal2/3",
            "VISal5",
            "VISpl4",
            "VISpl2/3",
            "VISpl5",
            "VISli4",
            "VISli2/3",
            "VISli5",
            "VISrl4",
            "VISrl2/3",
            "VISrl5",
            "VISl4",
            "VISl2/3",
            "VISl5",
            "VISpor4",
            "VISpor2/3",
            "VISpor5",
        ]

        if not isinstance(output_areas, list):
            self.output_areas = [output_areas]
        else:
            self.output_areas = output_areas

        for layer_conn in Conv_Params.keys():
            layer_sp = layer_conn.split("_")
            assert len(layer_sp) == 2
            curr_source = layer_sp[0]
            curr_target = layer_sp[1]
            self.Convs[layer_conn] = Conv2dMask(**Conv_Params[layer_conn])

            if curr_target not in self.BNs:
                self.BNs[curr_target] = nn.BatchNorm2d(**BN_Params[curr_target])

        # calculate total size output to classifier
        total_size = 0

        for area in self.output_areas:
            curr_source = "%s2/3" % area[:-1]
            curr_target = "%s" % area
            curr_out_channels = Conv_Params[curr_source + "_" + curr_target][
                "out_channels"
            ]
            if self.pool_size > 0:
                total_size += int(self.pool_size * self.pool_size * curr_out_channels)
            else:
                if curr_target in ["LGNv", "VISp4", "VISp2/3", "VISp5"]:
                    mult_factor = 64 * 64
                else:
                    mult_factor = 32 * 32
                total_size += int(mult_factor * curr_out_channels)

        if drop_final_fc:
            self.classifier = nn.Identity()
        else:
            self.classifier = nn.Sequential(nn.Linear(int(total_size), num_classes))

    def _propagate_layer_features(self, x):
        """
        function for getting activations for each layer of the model
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        """
        if isinstance(x, dict):
            assert "input" in x.keys()
            x = x["input"]

        self.layers = {}

        for area in self._areas:
            if area == "input":
                continue

            if area == "LGNv":
                self.layers["LGNv"] = self.Convs["input_LGNv"](x)
                continue

            for layer_conn in Conv_Params.keys():
                layer_sp = layer_conn.split("_")
                assert len(layer_sp) == 2
                curr_source = layer_sp[0]
                curr_target = layer_sp[1]
                if curr_target == area:
                    if area not in self.layers:
                        self.layers[area] = self.Convs[layer_conn](
                            self.layers[curr_source]
                        )
                    else:
                        # sum subsequent source connections into that target
                        self.layers[area] = self.layers[area] + self.Convs[layer_conn](
                            self.layers[curr_source]
                        )

            self.layers[area] = nn.ReLU(inplace=True)(self.BNs[area](self.layers[area]))

    def get_img_feature(self, x):
        """
        function for getting activations from a list of output_areas for input x
        :param x: input image set Tensor with size (num_img, INPUT_SIZE[0], INPUT_SIZE[1], INPUT_SIZE[2])
        :return: if list length is 1, return the flattened activation of that area
                 if list length is >1, return concatenated flattened activation of the areas.
        """
        self._propagate_layer_features(x)

        re = None
        for area in self.output_areas:
            if self.pool_size > 0:
                curr_area_out = torch.nn.AdaptiveAvgPool2d(self.pool_size)(
                    self.layers[area]
                )
            else:
                curr_area_out = self.layers[area]

            if re is None:
                re = torch.flatten(curr_area_out, start_dim=1)
            else:
                re = torch.cat([torch.flatten(curr_area_out, start_dim=1), re], axis=1)
        return re

    def forward(self, x):
        if self.pool_size == 0:
            # our classifier input unit size calculations assumed input image was 64x64
            if isinstance(x, dict):
                assert "input" in x.keys()
                x = x["input"]
            assert (x.shape[-1] == 64) and (x.shape[-2] == 64)
        x = self.get_img_feature(x)
        x = self.classifier(x)
        return x


def shi_mousenet(pretrained=False, progress=True, **kwargs):
    r"""Shi et al. 2020 MouseNet model architecture from their
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf> paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output areas (str):
    """
    model = ShiMouseNet(**kwargs)
    return model


def shi_mousenet_vispor5(pretrained=False, progress=True, **kwargs):
    r"""Variant of Shi et al. 2020 MouseNet model architecture
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf>
    that trains classifier on VISpor5 output only and with no avg pooling.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShiMouseNet(output_areas=["VISpor5"], pool_size=0, **kwargs)
    return model


def shi_mousenet_vispor5_pool_4(pretrained=False, progress=True, **kwargs):
    r"""Variant of Shi et al. 2020 MouseNet model architecture
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf>
    that trains classifier on VISpor5 output only and with 4x4 avg pooling.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShiMouseNet(output_areas=["VISpor5"], pool_size=4, **kwargs)
    return model


def shi_mousenet_ir(pretrained=False, progress=True, **kwargs):
    r"""Shi et al. 2020 MouseNet model architecture from their
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf> paper.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
        output areas (str):
    """
    model = ShiMouseNet(drop_final_fc=True, **kwargs)
    return model


def shi_mousenet_vispor5_ir(pretrained=False, progress=True, **kwargs):
    r"""Variant of Shi et al. 2020 MouseNet model architecture
    <https://www.biorxiv.org/content/10.1101/2020.10.23.353151v1.full.pdf>
    that trains classifier on VISpor5 output only and with no avg pooling.
    Args:
        pretrained (bool): If True, returns a model pre-trained on ImageNet
    """
    model = ShiMouseNet(
        drop_final_fc=True, output_areas=["VISpor5"], pool_size=0, **kwargs
    )
    return model
