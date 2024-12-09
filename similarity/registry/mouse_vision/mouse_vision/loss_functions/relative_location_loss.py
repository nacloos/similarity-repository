import torch
import torch.nn as nn

from itertools import chain
from mouse_vision.loss_functions.necks import RelativeLocationNeck
from mouse_vision.loss_functions.heads import ClsHead
from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["RelativeLocationLoss"]


class RelativeLocationLoss(LossFunctionBase):
    """Relative patch location.
    Implementation of "Unsupervised Visual Representation Learning
    by Context Prediction (https://arxiv.org/abs/1505.05192)".
    Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/5e67129743ef093ffe87999f7953532602917379/openselfsup/models/relative_loc.py
    Args:
        model_output_dim      : (int) output dimension of model without FC layer.
        hidden_dim.           : (int) dimension of hidden layer.
                                Default: None, uses the model_output_dim.
        neck_output_dim         : (int) dimension of output embedding for neck.
                                Default: 4096.
        num_classes           : (float) Number of classes (patches-1).
                                Default: 8.
    """

    def __init__(self, model_output_dim, neck_output_dim=4096, num_classes=8):
        super(RelativeLocationLoss, self).__init__()
        self.neck = RelativeLocationNeck(
            in_channels=model_output_dim, out_channels=neck_output_dim
        )
        self.neck.init_weights(init_linear="normal")
        self.head = ClsHead(in_channels=neck_output_dim, num_classes=num_classes)
        self.head.init_weights(init_linear="normal", std=0.005)

    def trainable_parameters(self):
        return chain(self.neck.parameters(), self.head.parameters())

    def named_parameters(self):
        return chain(self.neck.named_parameters(), self.head.named_parameters())

    def forward(self, model, inp, patch_label, **kwargs):
        """Forward computation during training.
        Args:
            inp (Tensor): Input of two concatenated images (8 patches each) of shape (N, 8, 2C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            Loss.

        Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/5e67129743ef093ffe87999f7953532602917379/openselfsup/models/relative_loc.py#L60-L107
        """
        assert inp.dim() == 5, "Input must have 5 dims, got: {}".format(
            inp.dim()
        )  # N x 8 x 2C x H x W
        # reshape batch dimension to work with backbone
        inp = inp.view(
            inp.size(0) * inp.size(1), inp.size(2), inp.size(3), inp.size(4)
        )  # (8N)x(2C)xHxW
        patch_label = torch.flatten(patch_label)  # (8N)
        # each path (batch dimension) with its corresponding central patch
        inp1, inp2 = torch.chunk(inp, 2, dim=1)  # each is 8N x C x H x W
        x1 = model(inp1)
        x2 = model(inp2)
        x = [torch.cat((x1, x2), dim=1)]
        x = self.neck(x)
        outs = self.head(x)
        loss_inputs = [outs, patch_label]
        losses = self.head.loss(*loss_inputs)
        return losses


if __name__ == "__main__":
    from mouse_vision.models.imagenet_models import resnet18

    model = resnet50(drop_final_fc=True)
    loss_func = RelativeLocationLoss(model_output_dim=2048)

    inputs = torch.rand(20, 8, 6, 224, 224)
    patch_label = torch.ones(20, 8)
    loss = loss_func(model, inputs, patch_label)
    print(loss)
