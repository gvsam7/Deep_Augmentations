import torch
import torch.nn as nn
import torch.nn.functional as F
# from Pool.MedianPool import MedianPool2d
import math


def unpack_param_2d(param):
    try:
        p_H, p_W = param[0], param[1]
    except:
        p_H, p_W = param, param
    return p_H, p_W


def median_pool_2d(input, kernel_size, stride, padding, dilation=1):
    # Input should be 4D (BCHW)
    assert(input.dim() == 4)

    # Get input dimensions
    b_size, c_size, h_size, w_size = input.size()

    # Get input parameters
    k_H, k_W = unpack_param_2d(kernel_size)
    s_H, s_W = unpack_param_2d(stride)
    p_H, p_W = unpack_param_2d(padding)
    d_H, d_W = unpack_param_2d(dilation)

    # First we unfold all the (kernel_size x kernel_size)  patches
    unf_input = F.unfold(input, kernel_size, dilation, padding, stride)

    # Reshape it so that each patch is a column
    row_unf_input = unf_input.reshape(b_size, c_size, k_H*k_W, -1)

    # Apply median operation along the columns for each channel separately
    med_unf_input, med_unf_indexes = torch.median(row_unf_input, dim=2, keepdim=True)

    # Restore original shape
    out_W = math.floor(((w_size + (2 * p_W) - (d_W * (k_W - 1)) - 1) / s_W) + 1)
    out_H = math.floor(((h_size + (2 * p_H) - (d_H * (k_H - 1)) - 1) / s_H) + 1)

    return med_unf_input.reshape(b_size, c_size, out_H, out_W)


class MixPool(nn.Module):
    def __init__(self, kernel_size, stride, padding, alpha):
        super(MixPool, self).__init__()
        alpha = torch.FloatTensor([alpha])
        self.alpha = nn.Parameter(alpha)
        self.kernel_size = kernel_size
        self.stride = stride
        self.padding = padding

    def forward(self, x):
        x = self.alpha * F.max_pool2d(x, self.kernel_size, self.stride, self.padding) + (1 - self.alpha) * \
            F.avg_pool2d(x, self.kernel_size, self.stride, self.padding)
            # median_pool_2d(x, self.kernel_size, self.stride, self.padding)
            # MedianPool2d(x, self.kernel_size, self.stride, self.padding)
        return x

def mixed(self):
    print("You are using Mixed Pooling Method")
    return MixPool


