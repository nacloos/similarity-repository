import numpy as np
import torch.nn as nn

from mouse_vision.model_training.custom_heads.custom_head_base import CustomHeadBase

__all__ = ["LinearReadout"]

class LinearReadout(CustomHeadBase):
    def __init__(self, model_output_dim=512, num_classes=1000):
        super(LinearReadout, self).__init__()

        self.model_output_dim = model_output_dim
        self.num_classes = num_classes

        # Initialize classifier head for transfer learning
        self.initialize_classifier()

    def initialize_classifier(self):
        assert hasattr(self, "classifier")

        self.classifier.add_module(
            "fc", nn.Sequential(
                    nn.Flatten(start_dim=1),
                    nn.Linear(self.model_output_dim, self.num_classes)
                )
        )

if __name__ == "__main__":
    import torch

    kwargs = {"model_output_dim": 1024, "num_classes": 1000}
    l = LinearReadout(**kwargs)
    data = torch.rand(10, 1024)
    output = l(data)
    print(output.shape)
    print(l.classifier)

