import torch
from torch import nn
from Gabor.GaborLayer import GaborConv2d
from Pool.MixPool import MixPool
from Pool.GatedPool import GatedPool_c, GatedPool_l
from Pool.SpatialPyramidPool import SPP


class GaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes, num_level=3, pool_type='average_pool'):
        super(GaborCNN, self).__init__()
        self.features = nn.Sequential(
            GaborConv2d(in_channels, out_channels=32, kernel_size=(3, 3)),
            nn.ReLU(inplace=True),
            # GatedPool_l(kernel_size=2, stride=2, padding=0),
            # GatedPool_c(in_channels=32, kernel_size=2, stride=2, padding=0),
            MixPool(2, 2, 0, 1),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(32),
            nn.Conv2d(32, 64, (3, 3)),
            nn.ReLU(inplace=True),
            # GatedPool_l(kernel_size=2, stride=2, padding=0),
            # GatedPool_c(in_channels=64, kernel_size=2, stride=2, padding=0),
            MixPool(2, 2, 0, 0.6),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(64),
            nn.Conv2d(64, 128, (3, 3)),
            nn.ReLU(inplace=True),
            # GatedPool_l(kernel_size=2, stride=2, padding=0),
            # GatedPool_c(in_channels=128, kernel_size=2, stride=2, padding=0),
            MixPool(2, 2, 0, 0.2),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(128),
            nn.Conv2d(128, 256, (3, 3)),
            nn.ReLU(inplace=True),
            # GatedPool_l(kernel_size=2, stride=2, padding=0),
            # GatedPool_c(in_channels=256, kernel_size=2, stride=2, padding=0),
            MixPool(2, 2, 0, 0.2),
            # nn.MaxPool2d(2),
            nn.BatchNorm2d(256),
            nn.Conv2d(256, 512, (3, 3)),
            nn.ReLU(inplace=True)
        )

        # self.avgpool = SPP(num_level)
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
