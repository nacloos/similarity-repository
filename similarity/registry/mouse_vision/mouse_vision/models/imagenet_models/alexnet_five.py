import torch
import torch.nn as nn

__all__ = ["AlexNetFive", "alexnet_five_64x64"]


class AlexNetFive(nn.Module):
    def __init__(self, num_classes=1000, pool_size=6):
        super(AlexNetFive, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(256 * pool_size * pool_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_five_64x64(pretrained=False, **kwargs):
    """
    AlexNet-based architecture with only five convolutional layers and
    a single fully-connected layer for classification.
    """
    model = AlexNetFive(**kwargs)

    return model


if __name__ == "__main__":
    m = alexnet_five_64x64()
    inputs = torch.rand(4, 3, 64, 64)
    outputs = m(inputs)
    print(outputs.shape)
