import torch.nn as nn
from models.Block import block_18
from Gabor.GaborLayer import GaborConv2d


class ResNet(nn.Module):
    def __init__(self, block_18, layers, in_channels, num_classes):
        super(ResNet, self).__init__()
        self.in_channels = 64
        self.conv1 = GaborConv2d(in_channels, 64, kernel_size=7, stride=2, padding=3)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU()
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        # ResNet layers
        self.layer1 = self._make_layer(block_18, layers[0], out_channels=64, stride=1)
        self.layer2 = self._make_layer(block_18, layers[1], out_channels=128, stride=2)
        self.layer3 = self._make_layer(block_18, layers[2], out_channels=256, stride=2)
        self.layer4 = self._make_layer(block_18, layers[3], out_channels=512, stride=2)  # 512

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = x.reshape(x.shape[0], -1)
        x = self.fc(x)

        return x

    def _make_layer(self, block_18, num_residual_blocks, out_channels, stride):
        identity_downsample = None
        layers = []

        if stride != 1 or self.in_channels != out_channels * 1:
            identity_downsample = nn.Sequential(
                nn.Conv2d(self.in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels),
            )

        layers.append(block_18(self.in_channels, out_channels, identity_downsample, stride))
        self.in_channels = out_channels  # 64*1=64

        for i in range(num_residual_blocks - 1):
            layers.append(block_18(self.in_channels, out_channels))  # 256 -> 64, o/p=64*4

        return nn.Sequential(*layers)


def GabResNet18(in_channels, num_classes=10):
    return ResNet(block_18, [2, 2, 2, 2], in_channels, num_classes)
