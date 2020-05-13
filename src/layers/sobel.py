import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_sobel_kernel_1d(channels):
    sobel_x = torch.tensor([-2, 0, +2], dtype=torch.float32).view(1, 1, 3)

    sobel_kernel = sobel_x.repeat(channels, 1, 1)

    return sobel_kernel


def get_sobel_kernel_2d(channels):
    sobel_x = torch.tensor([[-1, 0, +1],
                            [-2, 0, +2],
                            [-1, 0, +1]], dtype=torch.float32).view(1, 1, 3, 3)

    sobel_y = sobel_x.permute(0, 1, 3, 2)

    sobel_kernel = torch.cat([sobel_x, sobel_y], dim=0).repeat(channels, 1, 1, 1)

    return sobel_kernel


class Sobel(nn.Module):
    def __init__(self, channels, dim=2, n_pass=1):
        super(Sobel, self).__init__()

        self.groups = channels
        self.n_pass = n_pass

        if dim == 1:
            self.conv = F.conv1d
            self.register_buffer('weight', get_sobel_kernel_1d(channels))
        elif dim == 2:
            self.conv = F.conv2d
            self.register_buffer('weight', get_sobel_kernel_2d(channels))
        else:
            raise RuntimeError(
                'Only 1 and 2 dimensions are supported. Received {}.'.format(dim)
            )

    def forward(self, input):
        s_input = [input]
        for i in range(1, self.n_pass + 1):
            s_input.append(self.conv(s_input[i-1], weight=self.weight, stride=1, padding=1, groups=self.groups))
        return torch.cat(s_input, dim=1)


def get_sin_sobel_kernel_nd(channels, kernel_size, dim):
    sin_space = np.linspace(-np.pi / 2, np.pi / 2, kernel_size)
    sin_sobel = np.sin(sin_space) * 2
    sin_sobel = torch.tensor(sin_sobel, dtype=torch.float32).view(*([1, 1, kernel_size] + [1 for _ in range(dim - 1)]))
    if dim > 1:
        cos_space = np.linspace(-np.pi / 3, np.pi / 3, kernel_size)
        cos_sobel = np.cos(cos_space)
        cos_sobel = torch.tensor(cos_sobel, dtype=torch.float32).view(*([1, 1, 1, kernel_size] + [1 for _ in range(dim - 2)]))
        cos_sobel = cos_sobel.repeat(*([1, 1, 1, 1] + [kernel_size for _ in range(dim - 2)]))
        sin_sobel = sin_sobel * cos_sobel

        sin_sobel_l = [sin_sobel]
        for i in range(3, dim + 2):
            sin_sobel_l.append(sin_sobel.transpose(2, i))
        sin_sobel = torch.cat(sin_sobel_l, dim=0)

    sobel_kernel = sin_sobel.repeat(*([channels, 1] + [1 for _ in range(dim)]))

    return sobel_kernel


class SinSobel(nn.Module):
    def __init__(self, channels, kernel_size, padding, dim=2, n_pass=1):
        super(SinSobel, self).__init__()

        self.register_buffer('weight', get_sin_sobel_kernel_nd(channels, kernel_size, dim))
        self.groups = channels
        self.n_pass = n_pass
        self.padding = padding
        self.dim = dim

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
        s_input = [input]
        for i in range(1, self.n_pass + 1):
            s_input.append(self.conv(s_input[i-1], weight=self.weight, stride=1, padding=self.padding, groups=self.groups))
        return torch.cat(s_input, dim=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import imageio


    def close_event():
        plt.close()

    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(close_event)

    c_size = 128
    n_calls = 64

    canvas = torch.zeros([1, 1, c_size, c_size])
    canvas[:, :, c_size // 2, c_size // 2] = 1.0

    plt.imshow(canvas.view(c_size, c_size))
    timer.start()
    plt.show()
    canvas_l = [canvas.view(c_size, c_size)]

    sobel_f, pad_f = get_sobel_kernel_2d(1), 1
    # sobel_f, pad_f = get_sin_sobel_kernel_nd(1, 7, 2), 3

    for _ in range(n_calls):
        canvas_sob = F.conv2d(canvas, weight=sobel_f, stride=1, padding=pad_f)
        canvas = torch.cat([canvas, canvas_sob], dim=1)
        canvas = F.instance_norm(canvas)
        canvas = canvas.mean(dim=1, keepdim=True)
        plt.imshow(canvas.view(c_size, c_size))
        timer.start()
        plt.show()
        canvas_l.append(canvas.view(c_size, c_size))

    timer.stop()
    plt.imshow(canvas.view(c_size, c_size))
    plt.show()

    def _canvas(idx):
        return canvas_l[idx]

    imageio.mimsave('./sobel_waves.gif', [_canvas(i) for i in range(n_calls)], fps=5)
