import torch
import torch.nn as nn
import torch.distributed as dist

# from functools import partial
# from mouse_vision.models.custom_ops import SyncBatchNorm


def l2_normalize(x, dim=1):
    """
    Normalizes a set of vectors along dim so that the L2-norm is one.

    Inputs:
        x      : (torch.Tensor) vectors to normalize
        dim    : (int) dimension along which to normalize. Default: 1.

    Outputs:
        norm_x : (torch.Tensor) normalized vectors along dim
    """
    assert x.ndim == 2
    norm_x = x / torch.sqrt(torch.sum(x ** 2, dim=dim).unsqueeze(dim))
    return norm_x


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/aa62006c6e0fb3ee9474dbe8e009b65af35e8e06/openselfsup/models/utils/gather_layer.py
    """

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


# Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/aa62006c6e0fb3ee9474dbe8e009b65af35e8e06/openselfsup/models/utils/norm.py

norm_cfg = {
    # format: layer_type: (abbreviation, module)
    "BN": ("bn", nn.BatchNorm2d),
    "SyncBN": ("bn", nn.SyncBatchNorm),
    #    'SyncBN': ('bn', SyncBatchNorm),
    #    'TPUSyncBN': ('bn', partial(SyncBatchNorm, tpu=True)),
    "GN": ("gn", nn.GroupNorm),
    # and potentially 'SN'
}


def build_norm_layer(cfg, num_features, postfix=""):
    """Build normalization layer.
    Args:
        cfg (dict): cfg should contain:
            type (str): identify norm layer type.
            layer args: args needed to instantiate a norm layer.
            requires_grad (bool): [optional] whether stop gradient updates
        num_features (int): number of channels from input.
        postfix (int, str): appended into norm abbreviation to
            create named layer.
    Returns:
        name (str): abbreviation + postfix
        layer (nn.Module): created norm layer
    """
    if cfg is None:
        return "identity", nn.Identity()
    else:
        assert isinstance(cfg, dict) and "type" in cfg
        cfg_ = cfg.copy()

        layer_type = cfg_.pop("type")
        if layer_type not in norm_cfg:
            raise KeyError("Unrecognized norm type {}".format(layer_type))
        else:
            abbr, norm_layer = norm_cfg[layer_type]
            if norm_layer is None:
                raise NotImplementedError

        assert isinstance(postfix, (int, str))
        name = abbr + str(postfix)

        requires_grad = cfg_.pop("requires_grad", True)
        cfg_.setdefault("eps", 1e-5)
        if layer_type != "GN":
            layer = norm_layer(num_features, **cfg_)
            if layer_type == "SyncBN":
                # Note: if TPUSyncBN is ever supported, this is likely not relevant
                layer._specify_ddp_gpu_num(1)
        else:
            assert "num_groups" in cfg_
            layer = norm_layer(num_channels=num_features, **cfg_)

        for param in layer.parameters():
            param.requires_grad = requires_grad

        return name, layer


# Taken from: https://github.com/open-mmlab/mmcv/blob/63b7aa31b66796bdbdeaf179c39050dd6561de25/mmcv/cnn/utils/weight_init.py


def normal_init(module, mean=0, std=1, bias=0):
    nn.init.normal_(module.weight, mean, std)
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def kaiming_init(
    module, a=0, mode="fan_out", nonlinearity="relu", bias=0, distribution="normal"
):
    assert distribution in ["uniform", "normal"]
    if distribution == "uniform":
        nn.init.kaiming_uniform_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    else:
        nn.init.kaiming_normal_(
            module.weight, a=a, mode=mode, nonlinearity=nonlinearity
        )
    if hasattr(module, "bias") and module.bias is not None:
        nn.init.constant_(module.bias, bias)


def _init_weights(module, init_linear="normal", std=0.01, bias=0.0):
    assert init_linear in ["normal", "kaiming"], "Undefined init_linear: {}".format(
        init_linear
    )
    for m in module.modules():
        if isinstance(m, nn.Linear):
            if init_linear == "normal":
                normal_init(m, std=std, bias=bias)
            else:
                kaiming_init(m, mode="fan_in", nonlinearity="relu")
        elif isinstance(
            m, (nn.BatchNorm1d, nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)
        ):
            if m.weight is not None:
                nn.init.constant_(m.weight, 1)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
