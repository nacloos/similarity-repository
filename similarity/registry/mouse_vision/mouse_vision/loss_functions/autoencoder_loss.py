import torch
import torch.nn as nn
import numpy as np

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["AutoEncoderLoss"]


class AutoEncoderLoss(LossFunctionBase):
    def __init__(self, reduction="mean", l1_weighting=1e-4):
        super(AutoEncoderLoss, self).__init__()
        self.reduction = reduction
        self.output_loss = nn.MSELoss(reduction=self.reduction)
        self.l1_weighting = l1_weighting
        print(f"Using L1 weighting of {self.l1_weighting}")

    def forward(self, model, inp, **kwargs):
        preds = model(inp)

        l2_loss = self.output_loss(preds["output"], inp)
        l1_reg = torch.sum(torch.abs(preds["hidden_vec"]))
        loss = (
            0.5 * l2_loss
            + (self.l1_weighting / np.prod(preds["hidden_vec"].shape)) * l1_reg
        )
        return loss
