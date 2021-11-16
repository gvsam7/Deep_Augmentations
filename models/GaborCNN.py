import torch
from torch import nn
from Gabor.GaborLayer import GaborConv2d
from Pool.MixPool import MixPool


class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GaborCNN, self).__init__()
        self.features = nn.Sequential(
            GaborConv2d(in_channels, out_channels=32, kernel_size=(11, 11)),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 1),
            # nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.8),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.6),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            MixPool(2, 2, 0, 0.2),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True)
        )

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))

        self.classifier = nn.Sequential(
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = self.avgpool(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x