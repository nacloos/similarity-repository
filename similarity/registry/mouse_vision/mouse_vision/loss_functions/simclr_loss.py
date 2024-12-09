import torch
import torch.nn as nn

from mouse_vision.loss_functions.loss_utils import GatherLayer
from mouse_vision.loss_functions.necks import NonLinearNeckSimCLR
from mouse_vision.loss_functions.heads import ContrastiveHead
from mouse_vision.loss_functions.loss_function_base import LossFunctionBase

__all__ = ["SimCLRLoss"]


class SimCLRLoss(LossFunctionBase):
    """SimCLR.
    Implementation of "A Simple Framework for Contrastive Learning
    of Visual Representations (https://arxiv.org/abs/2002.05709)".
    Args:
        model_output_dim      : (int) output dimension of model without FC layer.
        hidden_dim.           : (int) dimension of hidden layer.
                                Default: None, uses the model_output_dim.
        embedding_dim         : (int) dimension of embedding for the hidden layer.
                                Default: 128.
        temperature           : (float) Temperature parameter of contrastive loss.
                                Default: 0.1.
    """

    def __init__(
        self,
        model_output_dim,
        hidden_dim=2048,
        embedding_dim=128,
        temperature=0.1,
        tpu=False,
    ):
        super(SimCLRLoss, self).__init__()

        self._hidden_dim = hidden_dim
        self.neck = NonLinearNeckSimCLR(
            in_channels=model_output_dim,
            hid_channels=self._hidden_dim,
            out_channels=embedding_dim,
            tpu=tpu,
        )
        self.neck.init_weights()
        self.head = ContrastiveHead(temperature=temperature)

    def trainable_parameters(self):
        return self.neck.parameters()

    def named_parameters(self):
        return self.neck.named_parameters()

    def _create_buffer(self, N, device):
        # ensures that these are on the same device as the input
        mask = 1 - torch.eye(N * 2, dtype=torch.uint8).to(device)
        pos_ind = (
            torch.arange(N * 2).to(device),
            2
            * torch.arange(N, dtype=torch.long)
            .unsqueeze(1)
            .repeat(1, 2)
            .view(-1, 1)
            .squeeze()
            .to(device),
        )
        neg_mask = torch.ones((N * 2, N * 2 - 1), dtype=torch.uint8).to(device)
        neg_mask[pos_ind] = 0
        return mask, pos_ind, neg_mask

    def forward(self, model, inp, **kwargs):
        """Forward computation during training.
        Args:
            inp (Tensor): Input of two concatenated images of shape (N, 2, C, H, W).
                Typically these should be mean centered and std scaled.
        Returns:
            Loss.

        Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/ed5000482b0d8b816cd8a6fbbb1f97da44916fed/openselfsup/models/simclr.py#L68-L96
        """
        assert inp.dim() == 5, "Input must have 5 dims, got: {}".format(inp.dim())
        # reshape batch dimension to work with backbone
        inp = inp.reshape(inp.size(0) * 2, inp.size(2), inp.size(3), inp.size(4))
        x = model(inp)  # 2n
        if not isinstance(x, list):
            x = [x]
        z = self.neck(x)[0]  # (2n)xd
        z = z / (torch.norm(z, p=2, dim=1, keepdim=True) + 1e-10)
        if inp.device.type == "xla":  # TPU
            # supports autodiff and does concatenation
            from torch_xla.core.functions import all_gather

            z = all_gather(z, dim=0)
        else:
            z = torch.cat(GatherLayer.apply(z), dim=0)  # (2N)xd
        assert z.size(0) % 2 == 0
        N = z.size(0) // 2
        s = torch.matmul(z, z.permute(1, 0))  # (2N)x(2N)
        mask, pos_ind, neg_mask = self._create_buffer(N, device=inp.device)
        # remove diagonal, (2N)x(2N-1)
        s = torch.masked_select(s, mask == 1).reshape(s.size(0), -1)
        positive = s[pos_ind].unsqueeze(1)  # (2N)x1
        # select negative, (2N)x(2N-2)
        negative = torch.masked_select(s, neg_mask == 1).reshape(s.size(0), -1)
        losses = self.head(positive, negative)
        return losses


if __name__ == "__main__":
    from mouse_vision.models.imagenet_models import resnet18

    model = resnet18(drop_final_fc=True)
    loss_func = SimCLRLoss(model_output_dim=512)

    inputs = torch.rand(20, 2, 3, 224, 224)
    loss = loss_func(model, inputs)
    print(loss)
