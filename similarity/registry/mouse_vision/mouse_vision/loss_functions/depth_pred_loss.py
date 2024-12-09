import torch.nn as nn

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase


__all__ = ["DepthPredictionHourGlassLoss"]


class DepthPredictionHourGlassLoss(LossFunctionBase):
    def __init__(self, output_channels=3):
        super(DepthPredictionHourGlassLoss, self).__init__()
        self.loss = nn.MSELoss(reduction="mean")
        self.decoder = nn.Conv2d(
            output_channels, 1, kernel_size=3, stride=1, padding=1, bias=False
        )

    def trainable_parameters(self):
        return self.decoder.parameters()

    def forward(self, x, output):
        depth_pred = self.decoder(x)
        loss = 0.5 * self.loss(depth_pred, output)
        return loss
