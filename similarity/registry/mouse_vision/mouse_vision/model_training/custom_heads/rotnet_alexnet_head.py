import numpy as np
import torch.nn as nn

from mouse_vision.model_training.custom_heads.custom_head_base import CustomHeadBase

__all__ = ["RotNetAlexNetHead"]

class RotNetAlexNetHead(CustomHeadBase):
    """
    Adapted from:
    https://github.com/gidariss/FeatureLearningRotNet/blob/master/architectures/LinearClassifier.py
    """
    def __init__(self, num_channels=256, pool_size=6, num_classes=1000):
        super(RotNetAlexNetHead, self).__init__()

        self.num_channels = num_channels
        self.pool_size = pool_size
        self.num_classes = num_classes

        # Initialize classifier head for transfer learning
        self.initialize_classifier()

    def initialize_classifier(self):
        assert hasattr(self, "classifier")

        total_features = self.num_channels * self.pool_size * self.pool_size

        self.classifier.add_module(
            "maxpool", nn.AdaptiveMaxPool2d((self.pool_size, self.pool_size))
        )
        self.classifier.add_module(
            "batchnorm", nn.BatchNorm2d(self.num_channels, affine=False)
        )
        self.classifier.add_module("flatten", nn.Flatten(start_dim=1))
        self.classifier.add_module(
            "linear_classifier", nn.Linear(total_features, self.num_classes)
        )
        self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                fin = m.in_features
                fout = m.out_features
                std_val = np.sqrt(2.0 / fout)
                m.weight.data.normal_(0.0, std_val)
                if m.bias is not None:
                    m.bias.data.fill_(0.0)

if __name__ == "__main__":
    import torch

    data = torch.rand(10,256,12,12)
    head = RotNetAlexNetHead(256, pool_size=6, num_classes=1000)
    output = head(data)
    print(output.shape)
    print(head.classifier)

