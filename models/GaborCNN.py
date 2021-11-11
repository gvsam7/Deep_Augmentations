import torch
from torch import nn
from Gabor.GaborLayer import GaborConv2d


class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes):
        super(GaborCNN, self).__init__()
        self.features = nn.Sequential(
            GaborConv2d(in_channels, out_channels=32, kernel_size=(11, 11)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2)
        )

        self.classifier = nn.Sequential(
            nn.Linear(12800, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, num_classes)
        )

    def forward(self, x):
        x = self.features(x)
        x = x.flatten(1)
        x = self.classifier(x)
        return x
