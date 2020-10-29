import math
import numbers
import torch
from torch import nn
from torch.nn import functional as F


class GaussianSmoothing(nn.Module):
    """
    Apply gaussian smoothing on a
    1d, 2d or 3d tensor. Filtering is performed seperately for each channel
    in the input using a depthwise convolution.
    Arguments:
        channels (int, sequence): Number of channels of the input tensors. Output will
            have this number of channels as well.
        kernel_size (int, sequence): Size of the gaussian kernel.
        sigma (float, sequence): Standard deviation of the gaussian kernel.
        dim (int, optional): The number of dimensions of the data.
            Default value is 2 (spatial).
    """
    def __init__(self, channels, kernel, padding, sigma, dim=2, left_side=False):
        super(GaussianSmoothing, self).__init__()

        self.register_buffer('weight', get_gaussian_kernel(channels, kernel, sigma, dim, left_side))
        self.groups = channels
        self.padding = padding

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        """
        Apply gaussian filter to input.
        Arguments:
            input (torch.Tensor): Input to apply gaussian filter on.
        Returns:
            filtered (torch.Tensor): Filtered output.
        """
        return self.conv(input, weight=self.weight, padding=self.padding, groups=self.groups)


class GaussianAttention(nn.Module):
    def __init__(self, channels, kernel=5, padding=2, sigma=1, dim=2, left_side=False, n_passes=4):
        super(GaussianAttention, self).__init__()

        self.register_buffer('weight', get_gaussian_kernel(channels, kernel, sigma, dim, left_side))
        self.groups = channels
        self.padding = padding
        self.n_passes = n_passes

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError(
                'Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, x):
        x_l = [x]
        for i in range(1, self.n_passes):
            x_l.append(self.conv(x_l[i-1], weight=self.weight, padding=self.padding, groups=self.groups))
        x_out = torch.cat(x_l, dim=1)
        return x_out


def get_gaussian_kernel(channels, kernel_size, sigma, dim=2, left_side=False):
    if isinstance(kernel_size, numbers.Number):
        kernel_size = [kernel_size] * dim
    if isinstance(sigma, numbers.Number):
        sigma = [sigma] * dim

    # The gaussian kernel is the product of the
    # gaussian function of each dimension.
    kernel = 1
    meshgrids = torch.meshgrid([torch.arange(size, dtype=torch.float32) for size in kernel_size])
    for size, std, mgrid in zip(kernel_size, sigma, meshgrids):
        mean = (size - 1) / 2
        new_kernel = 1 / (std * math.sqrt(2 * math.pi)) * torch.exp(-((mgrid - mean) / std) ** 2 / 2)
        if left_side:
            new_kernel[(size // 2) + 1:] = 0.
        kernel *= new_kernel

    if left_side:
        # Make sure the max value in the kernel is 1.
        kernel = kernel / torch.max(kernel)
    else:
        # Make sure sum of values in gaussian kernel equals 1.
        kernel = kernel / torch.sum(kernel)

    # Reshape to depthwise convolutional weight
    kernel = kernel.view(1, 1, *kernel.size())
    kernel = kernel.repeat(channels, *[1] * (kernel.dim() - 1))

    return kernel
