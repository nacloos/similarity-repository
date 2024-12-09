import torch.nn as nn

__all__ = ["CustomHeadBase"]

class CustomHeadBase(nn.Module):
    def __init__(self):
        super(CustomHeadBase, self).__init__()

        # Create the classifier head for transfer learning
        self.classifier = nn.Sequential()

    def initialize_classifier(self):
        raise NotImplementedError

    def forward(self, x):
        return self.classifier(x)

