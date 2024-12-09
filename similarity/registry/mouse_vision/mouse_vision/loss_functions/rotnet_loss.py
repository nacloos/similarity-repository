import torch.nn as nn

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["RotNetLoss"]


class RotNetLoss(LossFunctionBase):
    """
    This loss just wraps the cross-entropy loss but also incorporates the
    FC head for the rotation prediction

    Arguments:
        model_output_dim : (int) number of output dimensions of the model backbone
    """

    def __init__(self, model_output_dim):
        super(RotNetLoss, self).__init__()
        self.loss = nn.CrossEntropyLoss(reduction="mean")

        # Output is 4: prediction of four rotations
        self.fc = nn.Linear(model_output_dim, 4)

    def trainable_parameters(self):
        return self.fc.parameters()

    def forward(self, model, inputs, targets):
        """
        Main entry point for RotNet loss. It takes as input the batch output from
        the model backbone and the batch labels.

        Inputs:
            model   : (torch.nn.Module) model object
            inputs  : (torch.Tensor) input images and rotations. It should be of
                      dimensions: (N, 4, C, H, W) since there are 4 rotations
            targets : (torch.Tensor) batch labels. It should be of dimensions
                      (N, 4), the four labels for each of the rotations.
        """
        assert inputs.shape[1] == targets.shape[1] == 4

        # First flatten the inputs and labels
        inputs = inputs.reshape(
            inputs.shape[0] * 4, inputs.shape[2], inputs.shape[3], inputs.shape[4]
        )
        targets = targets.reshape(targets.shape[0] * 4)
        assert inputs.shape[0] == targets.shape[0]

        # Compute model outputs
        outputs = model(inputs)

        # Compute rotation predictions
        predictions = self.fc(outputs)

        # Compute the loss
        loss = self.loss(predictions, targets)

        return loss, predictions
