import torch
from torch import nn
from math import floor, ceil
import torch.nn.functional as F


class SPP(nn.Module):
    def __init__(self, num_level, pool_type='max_pool'):
        super(SPP, self).__init__()
        self.num_level = num_level
        self.pool_type = pool_type

    def forward(self, x):
        N, C, H, W =x.size()
        for i in range(self.num_level):
            level = i + 1
            kernel_size = (ceil(H/level), ceil(W/level))
            stride = (ceil(H/level), ceil(W/level))
            padding = (floor((kernel_size[0] * level - H + 1) / 2), floor((kernel_size[1] * level - W + 1) / 2))

            if self.pool_type == 'max_pool':
                tensor = (F.max_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
                # tensor = nn.MaxPool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)
            else:
                tensor = (F.avg_pool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)).view(N, -1)
                # tensor = nn.AvgPool2d(x, kernel_size=kernel_size, stride=stride, padding=padding)

            if i == 0:
                ssp = tensor
            else:
                ssp = torch.cat((ssp, tensor), 1)
        return ssp

    def __repr__(self):
        return self.__class__.__name__ + '('+ 'num_level = ' + str(self.num_level) + ', pool_type = ' + str(self.pool_type) + ')'
