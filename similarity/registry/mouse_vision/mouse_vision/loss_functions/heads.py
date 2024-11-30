import torch
import torch.nn as nn
from mouse_vision.loss_functions.loss_utils import kaiming_init, normal_init
from mouse_vision.model_training.train_utils import compute_accuracy


class ContrastiveHead(nn.Module):
    """Head for contrastive learning.
    Adapted from: https://github.com/open-mmlab/OpenSelfSup/blob/aa62006c6e0fb3ee9474dbe8e009b65af35e8e06/openselfsup/models/heads/contrastive_head.py#L8-L38
    Args:
        temperature (float): The temperature hyper-parameter that
            controls the concentration level of the distribution.
            Default: 0.1.
    """

    def __init__(self, temperature=0.1, return_accs=False):
        super(ContrastiveHead, self).__init__()
        self.criterion = nn.CrossEntropyLoss()
        self.temperature = temperature
        self.return_accs = return_accs

    def forward(self, pos, neg):
        """Forward head.
        Args:
            pos (Tensor): Nx1 positive similarity.
            neg (Tensor): Nxk negative similarity.
        Returns:
            Tensor: The loss.
        """
        N = pos.size(0)
        # logits: Nx(1+K)
        logits = torch.cat((pos, neg), dim=1)
        logits /= self.temperature
        # put labels on the same device as logits
        labels = torch.zeros((N,), dtype=torch.long).to(logits.device)
        loss = self.criterion(logits, labels)
        if self.return_accs:
            losses = dict()
            losses["loss"] = loss
            # we compute the accuracy here b/c in case the labels are created here
            acc1, acc5 = compute_accuracy(output=logits, target=labels, topk=(1, 5))
            losses["acc1"] = acc1
            losses["acc5"] = acc5
            return losses
        else:
            return loss


class ClsHead(nn.Module):
    """Simplest classifier head, with only one fc layer.
    Taken from: https://github.com/open-mmlab/OpenSelfSup/blob/e4f09cecf50a54540f7633190387ea060b1bc49e/openselfsup/models/heads/cls_head.py#L9-L53
    """

    def __init__(self, with_avg_pool=False, in_channels=2048, num_classes=1000):
        super(ClsHead, self).__init__()
        self.with_avg_pool = with_avg_pool
        self.in_channels = in_channels
        self.num_classes = num_classes

        self.criterion = nn.CrossEntropyLoss()

        if self.with_avg_pool:
            self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc_cls = nn.Linear(in_channels, num_classes)

    def init_weights(self, init_linear="normal", std=0.01, bias=0.0):
        assert init_linear in ["normal", "kaiming"], "Undefined init_linear: {}".format(
            init_linear
        )
        for m in self.modules():
            if isinstance(m, nn.Linear):
                if init_linear == "normal":
                    normal_init(m, std=std, bias=bias)
                else:
                    kaiming_init(m, mode="fan_in", nonlinearity="relu")
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm, nn.SyncBatchNorm)):
                if m.weight is not None:
                    nn.init.constant_(m.weight, 1)
                if m.bias is not None:
                    nn.init.constant_(m.bias, 0)

    def forward(self, x):
        assert isinstance(x, (tuple, list)) and len(x) == 1
        x = x[0]
        if self.with_avg_pool:
            assert x.dim() == 4, "Tensor must has 4 dims, got: {}".format(x.dim())
            x = self.avg_pool(x)
        x = x.view(x.size(0), -1)
        cls_score = self.fc_cls(x)
        return [cls_score]

    def loss(self, cls_score, labels):
        losses = dict()
        assert isinstance(cls_score, (tuple, list)) and len(cls_score) == 1
        losses["loss"] = self.criterion(cls_score[0], labels)
        # we compute the accuracy here b/c in case the labels (like for rel loc) they are modified from loader
        acc1, acc5 = compute_accuracy(output=cls_score[0], target=labels, topk=(1, 5))
        losses["acc1"] = acc1
        losses["acc5"] = acc5
        return losses
