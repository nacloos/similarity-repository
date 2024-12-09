import torch.nn as nn

from mouse_vision.loss_functions.loss_function_base import LossFunctionBase
from mouse_vision.loss_functions.necks import ProjectionMLPSimSiam, PredictionMLPSimSiam
from mouse_vision.loss_functions.loss_utils import l2_normalize


__all__ = ["SimSiamLoss"]


class SimSiamLoss(LossFunctionBase):
    """
    Implementation for simple siamese representation learning.
    Chen and He (2020): https://arxiv.org/pdf/2011.10566.pdf
    """

    def __init__(
        self,
        model_output_dim,
        projection_mlp_hidden_dim=2048,
        projection_mlp_output_dim=2048,
        prediction_mlp_hidden_dim=512,
        prediction_mlp_output_dim=2048,
    ):
        super(SimSiamLoss, self).__init__()

        self.projection_mlp = ProjectionMLPSimSiam(
            model_output_dim,
            hidden_dim=projection_mlp_hidden_dim,
            output_dim=projection_mlp_output_dim,
        )

        self.prediction_mlp = PredictionMLPSimSiam(
            input_dim=projection_mlp_output_dim,
            hidden_dim=prediction_mlp_hidden_dim,
            output_dim=prediction_mlp_output_dim,
        )

        self.neck = nn.ModuleDict(
            {
                "projection_mlp": self.projection_mlp,
                "prediction_mlp": self.prediction_mlp,
            }
        )

    def trainable_parameters(self):
        params = list(self.projection_mlp.parameters())
        params += list(self.prediction_mlp.parameters())
        return params

    def D(self, p, z):
        # D is defined on page 2, equation 1
        assert p.shape == z.shape
        assert p.ndim == 2

        z = z.detach()  # Stop gradient
        p = l2_normalize(p)
        z = l2_normalize(z)

        # Take inner product of p and z
        neg_cosine_sim = -1.0 * (p * z).sum(dim=1)

        # Return average loss across batch
        return neg_cosine_sim.mean()

    def forward(self, model, x1, x2):
        # x1 is first augmentation and x2 is second augmentation
        assert x1.ndim == x2.ndim == 4  # NCHW
        assert x1.shape == x2.shape

        f_x1 = self.projection_mlp(model(x1))  # z_1
        f_x2 = self.projection_mlp(model(x2))  # z_2

        h_x1 = self.prediction_mlp(f_x1)  # p_1
        h_x2 = self.prediction_mlp(f_x2)  # p_2

        # Page 3, equation 2
        loss = 0.5 * (self.D(h_x1, f_x2) + self.D(h_x2, f_x1))
        return loss
