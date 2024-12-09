import torch.nn as nn
import torch.nn.functional as F


class Encoder(nn.Module):

    def __init__(self, in_dim, out_dim, mid_layers):
        super(Encoder, self).__init__()
        layers = [nn.Linear(in_dim, mid_layers[0]), nn.ReLU()]
        for i, layer_dim in zip(range(1, len(mid_layers)), mid_layers[1:]):
            layers.append(nn.Linear(mid_layers[i - 1], layer_dim))
            layers.append(nn.ReLU())
        layers.append(nn.Linear(mid_layers[-1], out_dim))
        self.net = nn.Sequential(*layers)

    def forward(self, vector):
        return F.normalize(self.net(vector), dim=1)