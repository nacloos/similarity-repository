import torch.nn as nn
import torch
from cka_pytorch import feature_space_linear_cka_pytorch
from models import Encoder


class deepCKA(nn.Module):
    def __init__(self, in_dim, out_dim, mid_layers, debiased=False):
        super(deepCKA, self).__init__()
        self.encoder = Encoder(in_dim, out_dim, mid_layers)
        self.debiased = debiased

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        return feature_space_linear_cka_pytorch(x1, x2, debiased=self.debiased)

    def get_features(self, x):
        return self.encoder(x)


class deepDot(nn.Module):
    def __init__(self, in_dim, out_dim, mid_layers):
        super(deepDot, self).__init__()
        self.encoder = Encoder(in_dim, out_dim, mid_layers)

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        return torch.dot(x1.view(-1), x2.view(-1)) / x1.shape[0]

    def get_features(self, x):
        return self.encoder(x)


class ContrastiveSim(nn.Module):
    def __init__(self, in_dim, out_dim, mid_layers):
        super(ContrastiveSim, self).__init__()
        self.encoder = Encoder(in_dim, out_dim, mid_layers)

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        return torch.dot(x1.view(-1), x2.view(-1)) / x1.shape[0]

    def get_features(self, x):
        return self.encoder(x)


class ContrastiveSim_dis(nn.Module):
    def __init__(self, in_dim, out_dim, mid_layers):
        super(ContrastiveSim_dis, self).__init__()
        self.encoder = Encoder(in_dim, out_dim, mid_layers)

    def forward(self, x1, x2):
        x1, x2 = self.encoder(x1), self.encoder(x2)
        return (1 - (x1 - x2).norm(dim=-1, p=2)).mean()

    def get_features(self, x):
        return self.encoder(x)


if __name__ == '__main__':
    encoder = ContrastiveSim_dis(768, 512, [512])
    sim = encoder(torch.ones((10, 768)), torch.ones((10, 768)))
    pass
