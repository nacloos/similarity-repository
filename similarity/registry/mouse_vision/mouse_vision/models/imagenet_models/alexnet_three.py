import torch
import torch.nn as nn

__all__ = ["AlexNetThree", "alexnet_three_64x64"]


class AlexNetThree(nn.Module):
    def __init__(self, num_classes=1000, pool_size=6):
        super(AlexNetThree, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(64, 192, kernel_size=5, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),
            nn.Conv2d(192, 384, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
        )

        self.avgpool = nn.AdaptiveAvgPool2d((pool_size, pool_size))

        self.classifier = nn.Sequential(
            nn.Dropout(), nn.Linear(384 * pool_size * pool_size, num_classes),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.classifier(x)
        return x


def alexnet_three_64x64(pretrained=False, **kwargs):
    """
    AlexNet-based architecture with only three convolutional layers and
    a single fully-connected layer for classification.
    """
    model = AlexNetThree(**kwargs)

    return model


if __name__ == "__main__":
    m = alexnet_three_64x64()
    inputs = torch.rand(4, 3, 64, 64)
    outputs = m(inputs)
    print(outputs.shape)
