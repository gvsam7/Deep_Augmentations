import torch.nn as nn
from Gabor.GaborLayer import GaborConv2d


class DACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(DACBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=1)
        self.conv3 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=3)
        self.conv6 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=6)
        self.conv9 = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding='same', stride=1, dilation=9)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.conv1(x)
        x3 = self.conv3(x)
        x6 = self.conv6(x)
        x9 = self.conv9(x)
        x_t = x1 + (x3 + x6 + x9)/3
        return self.relu(self.bn(x_t))


class ACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(ACBlock, self).__init__()
        self.square = nn.Conv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.square(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.relu(self.bn(x1 + x2 + x3))


class GACBlock(nn.Module):
    def __init__(self, in_planes, out_planes):
        super(GACBlock, self).__init__()
        self.square = GaborConv2d(in_planes, out_planes, kernel_size=3, padding=1, stride=1)
        self.cross_ver = nn.Conv2d(in_planes, out_planes, kernel_size=(1, 3), padding=(0, 1), stride=1)
        self.cross_hor = nn.Conv2d(in_planes, out_planes, kernel_size=(3, 1), padding=(1, 0), stride=1)
        self.bn = nn.BatchNorm2d(out_planes)
        self.relu = nn.ReLU(True)

    def forward(self, x):
        x1 = self.square(x)
        x2 = self.cross_ver(x)
        x3 = self.cross_hor(x)
        return self.relu(self.bn(x1 + x2 + x3))