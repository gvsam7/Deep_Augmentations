import torch
from torch import nn
from Pool.MixPool import MixPool
from models.ConvBlock import DACBlock
from models.ConvBlock import ACBlock, GACBlock


class ACDilGaborCNN(nn.Module):
    def __init__(self, in_channels, num_classes, num_level=3, pool_type='average_pool'):
        super(ACDilGaborCNN, self).__init__()
        self.features = nn.Sequential(
            GACBlock(in_channels, 32),
            MixPool(2, 2, 0, 1),
            ACBlock(32, 64),
            MixPool(2, 2, 0, 0.6),
            ACBlock(64, 128),
            MixPool(2, 2, 0, 0.2),
            ACBlock(128, 256),
            MixPool(2, 2, 0, 0.2),
            ACBlock(256, 512),
            DACBlock(512, 512)
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
