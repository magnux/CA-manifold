import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_dither_kernel_1d(channels):
    dither_x = torch.tensor([1/8, 0, 1/8], dtype=torch.float32).view(1, 1, 3)

    dither_kernel = dither_x.repeat(channels, 1, 1)

    return dither_kernel


def get_dither_kernel_2d(channels):
    dither_x = torch.tensor([[1/16, 1/16, 1/16],
                             [1/16, 0.,   1/16],
                             [1/16, 1/16, 1/16]], dtype=torch.float32).view(1, 1, 3, 3)

    dither_y = dither_x.permute(0, 1, 3, 2)

    dither_kernel = torch.cat([dither_x, dither_y], dim=0).repeat(channels, 1, 1, 1)

    return dither_kernel


class ChannelDither(nn.Module):
    def __init__(self, channels, dim=2):
        super(ChannelDither, self).__init__()

        self.groups = channels

        if dim == 1:
            self.conv = F.conv1d
            self.register_buffer('weight', get_dither_kernel_1d(channels))
        elif dim == 2:
            self.conv = F.conv2d
            self.register_buffer('weight', get_dither_kernel_2d(channels))
        else:
            raise RuntimeError(
                'Only 1 and 2 dimensions are supported. Received {}.'.format(dim)
            )
        self.c_factor = 1 + dim

    def forward(self, x):
        x_dither = self.conv(F.relu(-x), weight=self.weight, stride=1, padding=1, groups=self.groups)
        return x + x_dither
