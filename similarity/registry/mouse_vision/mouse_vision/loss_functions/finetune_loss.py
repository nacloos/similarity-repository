import torch.nn as nn

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase
from mouse_vision.model_training.custom_heads import CustomHeadBase

__all__ = ["FinetuneLoss"]


class FinetuneLoss(LossFunctionBase):
    """
    This loss just wraps the cross-entropy loss but also incorporates the
    FC head.
    
    Arguments:
        readout_module : (CustomHeadBase) an instance of nn.Module that is the
                         custom head readout for finetuning / transfer
    """

    def __init__(self, readout_module):
        super(FinetuneLoss, self).__init__()
        assert isinstance(readout_module, CustomHeadBase)

        self.loss = nn.CrossEntropyLoss(reduction="mean")
        self.readout = readout_module

    def trainable_parameters(self):
        return self.readout.parameters()

    def forward(self, outputs, targets):
        """
        Main entry point for finetune loss. It takes as input the batch output from
        the model backbone and the batch labels.

        Inputs:
            outputs : (torch.Tensor) outputs of the model backbone
            targets : (torch.Tensor) batch labels
        """
        # First past the backbone outputs through a readout layer
        preds = self.readout(outputs)

        # Compute the loss
        loss = self.loss(preds, targets)

        return loss, preds
