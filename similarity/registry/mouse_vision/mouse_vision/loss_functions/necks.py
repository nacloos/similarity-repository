import torch
import torch.nn as nn

from mouse_vision.loss_functions.loss_utils import build_norm_layer, _init_weights


class NonLinearNeckSimCLR(nn.Module):
    """SimCLR non-linear neck.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/aa62006c6e0fb3ee9474dbe8e009b65af35e8e06/openselfsup/models/necks.py#L234-L336
    Structure: fc(no_bias)-bn(has_bias)-[relu-fc(no_bias)-bn(no_bias)].
        The substructures in [] can be repeated. For the SimCLR default setting,
        the repeat time is 1.
    However, PyTorch does not support to specify (weight=True, bias=False).
        It only support \"affine\" including the weight and bias. Hence, the
        second BatchNorm has bias in this implementation. This is different from
        the official implementation of SimCLR.
    Since SyncBatchNorm in pytorch<1.4.0 does not support 2D input, the input is
        expanded to 4D with shape: (N,C,1,1). Not sure if this workaround
        has no bugs. See the pull request here:
        https://github.com/pytorch/pytorch/pull/29626.
    Args:
        num_layers (int): Number of fc layers, it is 2 in the SimCLR default setting.
    """

    def __init__(
        self,
        in_channels,
        hid_channels,
        out_channels=128,
        num_layers=2,
        sync_bn=True,
        with_bias=False,
        with_last_bn=True,
        with_avg_pool=False,
        tpu=False,
    ):
        super(NonLinearNeckSimCLR, self).__init__()
        self.sync_bn = sync_bn
        self.with_last_bn = with_last_bn
        self.with_avg_pool = with_avg_pool
        self.tpu = tpu
        #        self.sync_bn_name = 'TPUSyncBN' if self.tpu else 'SyncBN'
        # uncomment the above and comment the below 3 lines if TPUSyncBN is supported
        self.sync_bn_name = "SyncBN"
        if self.tpu:
            self.sync_bn = False

        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if torch.__version__ < "1.4.0":
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        self.relu = nn.ReLU(inplace=True)
        self.fc0 = nn.Linear(in_channels, hid_channels, bias=with_bias)
        if self.sync_bn:
            _, self.bn0 = build_norm_layer(dict(type=self.sync_bn_name), hid_channels)
        else:
            self.bn0 = nn.BatchNorm1d(hid_channels)

        self.fc_names = []
        self.bn_names = []
        for i in range(1, num_layers):
            this_channels = out_channels if i == num_layers - 1 else hid_channels
            self.add_module(
                "fc{}".format(i), nn.Linear(hid_channels, this_channels, bias=with_bias)
            )
            self.fc_names.append("fc{}".format(i))
            if i != num_layers - 1 or self.with_last_bn:
                if self.sync_bn:
                    self.add_module(
                        "bn{}".format(i),
                        build_norm_layer(dict(type=self.sync_bn_name), this_channels)[
                            1
                        ],
                    )
                else:
                    self.add_module("bn{}".format(i), nn.BatchNorm1d(this_channels))
                self.bn_names.append("bn{}".format(i))
            else:
                self.bn_names.append(None)

    def init_weights(self, init_linear="normal"):
        _init_weights(self, init_linear)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc0(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn0, x)
        else:
            x = self.bn0(x)
        for fc_name, bn_name in zip(self.fc_names, self.bn_names):
            fc = getattr(self, fc_name)
            x = self.relu(x)
            x = fc(x)
            if bn_name is not None:
                bn = getattr(self, bn_name)
                if self.sync_bn:
                    x = self._forward_syncbn(bn, x)
                else:
                    x = bn(x)
        return [x]


class RelativeLocationNeck(nn.Module):
    """Relative patch location neck: fc-bn-relu-dropout.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/7f071cd15c56f8de7a3a16f1e55b285b765f1e99/openselfsup/models/necks.py#L51-L106
    """

    def __init__(self, in_channels, out_channels, sync_bn=False, with_avg_pool=False):
        super(RelativeLocationNeck, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        if torch.__version__ < "1.4.0":
            self.expand_for_syncbn = True
        else:
            self.expand_for_syncbn = False

        # since we have two patches, stacked in the channel dimension
        self.fc = nn.Linear(in_channels * 2, out_channels)
        if sync_bn:
            _, self.bn = build_norm_layer(
                dict(type="SyncBN", momentum=0.003), out_channels
            )
        else:
            self.bn = nn.BatchNorm1d(out_channels, momentum=0.003)
        self.relu = nn.ReLU(inplace=True)
        self.drop = nn.Dropout()
        self.sync_bn = sync_bn

    def init_weights(self, init_linear="normal"):
        _init_weights(self, init_linear, std=0.005, bias=0.1)

    def _forward_syncbn(self, module, x):
        assert x.dim() == 2
        if self.expand_for_syncbn:
            x = module(x.unsqueeze(-1).unsqueeze(-1)).squeeze(-1).squeeze(-1)
        else:
            x = module(x)
        return x

    def forward(self, x):
        assert len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            x = self.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.sync_bn:
            x = self._forward_syncbn(self.bn, x)
        else:
            x = self.bn(x)
        x = self.relu(x)
        x = self.drop(x)
        return [x]


class NonLinearNeckMoCov2(nn.Module):
    """The non-linear neck in MoCo v2: fc-relu-fc.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/7f071cd15c56f8de7a3a16f1e55b285b765f1e99/openselfsup/models/necks.py#L174-L199
    """

    def __init__(self, in_channels, hid_channels, out_channels, with_avg_pool=False):
        super(NonLinearNeckMoCov2, self).__init__()
        self.with_avg_pool = with_avg_pool
        if with_avg_pool:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.mlp = nn.Sequential(
            nn.Linear(in_channels, hid_channels),
            nn.ReLU(inplace=True),
            nn.Linear(hid_channels, out_channels),
        )

    def init_weights(self, init_linear="normal"):
        _init_weights(self, init_linear)

    def forward(self, x):
        if self.with_avg_pool:
            x = self.avgpool(x)
        return [self.mlp(x.view(x.size(0), -1))]


class ProjectionMLPSimSiam(nn.Module):
    """
    See page 3 bottom of https://arxiv.org/pdf/2011.10566.pdf
    """

    assert torch.__version__ > "1.4.0"  # For SyncBN

    def __init__(
        self, model_output_dim, hidden_dim=2048, output_dim=2048, sync_bn=True
    ):
        super(ProjectionMLPSimSiam, self).__init__()

        if sync_bn:
            _, bn1 = build_norm_layer(dict(type="SyncBN"), hidden_dim)
            _, bn2 = build_norm_layer(dict(type="SyncBN"), hidden_dim)
            _, bn3 = build_norm_layer(dict(type="SyncBN"), output_dim)
        else:
            bn1 = nn.BatchNorm1d(hidden_dim)
            bn2 = nn.BatchNorm1d(hidden_dim)
            bn3 = nn.BatchNorm1d(output_dim)

        self.layer1 = nn.Sequential(
            nn.Linear(model_output_dim, hidden_dim), bn1, nn.ReLU(),
        )

        self.layer2 = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), bn2, nn.ReLU(),)

        self.layer3 = nn.Sequential(nn.Linear(hidden_dim, output_dim), bn3)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)

        return x


class PredictionMLPSimSiam(nn.Module):
    """
    See page 3 bottom of https://arxiv.org/pdf/2011.10566.pdf
    """

    assert torch.__version__ > "1.4.0"  # For SyncBN

    def __init__(self, input_dim=2048, hidden_dim=512, output_dim=2048, sync_bn=True):
        super(PredictionMLPSimSiam, self).__init__()

        if sync_bn:
            _, bn = build_norm_layer(dict(type="SyncBN"), hidden_dim)
        else:
            bn = nn.BatchNorm1d(hidden_dim)

        # "The dimension of h’s input and output (z and p) is d =
        # 2048, and h’s hidden layer’s dimension is 512, making h
        # a bottleneck structure." (page 3 of paper)
        self.layer1 = nn.Sequential(nn.Linear(input_dim, hidden_dim), bn, nn.ReLU(),)

        self.layer2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = self.layer1(x)
        x = self.layer2(x)

        return x
