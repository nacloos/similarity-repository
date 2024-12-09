import torch
import torch.nn as nn
import numpy as np

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["CrossEntropyLoss"]


class CrossEntropyLoss(LossFunctionBase):
    def __init__(self, reduction="mean"):
        super(CrossEntropyLoss, self).__init__()
        self.reduction = reduction
        self.loss = nn.CrossEntropyLoss(reduction=self.reduction)

    def forward(self, model, inp, target, **kwargs):
        preds = model(inp)
        loss = self.loss(preds, target)
        return loss, preds


if __name__ == "__main__":
    c = CrossEntropyLoss(mean=[0, 0, 0], std=[1, 1, 1])
