import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_dither_kernel_1d(channels):
    dither_kernel = 1e-2 * torch.randn((1, 1, 3), dtype=torch.float32)
    dither_kernel[:, :, 1] = 0.

    dither_kernel = dither_kernel.repeat(channels, 1, 1)

    return dither_kernel


def get_dither_kernel_2d(channels):
    dither_kernel = 1e-2 * torch.randn((1, 1, 3, 3), dtype=torch.float32)
    dither_kernel[:, :, 1, 1] = 0.

    dither_kernel = dither_kernel.repeat(channels, 1, 1, 1)

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
        x_dither = self.conv(x, weight=self.weight, stride=1, padding=1, groups=self.groups)
        return x + x_dither


if __name__ == '__main__':
    import matplotlib.pyplot as plt


    def close_event():
        plt.close()

    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(close_event)

    c_size = 64
    # c_size = 128

    canvas = torch.Tensor(np.sum(np.indices((c_size, c_size)), axis=0)).reshape(1, 1, c_size, c_size)
    canvas = (2 * (canvas/ canvas.max())) - 1

    plt.imshow(canvas.view(c_size, c_size))
    plt.show()

    plt.imshow(F.relu(canvas).view(c_size, c_size))
    plt.show()

    cd = ChannelDither(1)
    plt.imshow(cd(canvas).view(c_size, c_size) - canvas.view(c_size, c_size))
    plt.show()

    dit_diff = cd(canvas) - canvas
    print(dit_diff.mean(), dit_diff.std(), dit_diff.max(), dit_diff.min())
