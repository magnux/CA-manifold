import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_circle_grads_kernel_nd(channels, kernel_size, dim):
    circle_grads = torch.zeros([kernel_size for _ in range(dim)])
    for i in range(dim - 1):
        x = torch.linspace(-1.0, 1.0, kernel_size).view(*([kernel_size if i == d else 1 for d in range(dim)]))
        x = x.repeat(*([1 if i == d else kernel_size for d in range(dim)]))
        for j in range(i + 1, dim):
            y = torch.linspace(-1.0, 1.0, kernel_size).view(*([kernel_size if j == d else 1 for d in range(dim)]))
            y = y.repeat(*([1 if j == d else kernel_size for d in range(dim)]))
            circle_grads[(x * x + y * y > 0.9) * (x * x + y * y < 1.3)] = 1.
            circle_grads[(x * x + y * y > 0.5) * (x * x + y * y < 0.9)] = -1.

    circle_grads = circle_grads.to(dtype=torch.float32).unsqueeze(0).unsqueeze(1)
    circle_grads_kernel = circle_grads.repeat(*([channels, 1] + [1 for _ in range(dim)]))

    return circle_grads_kernel


class CircleGrads(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings, dim=2, mode='split_out'):
        super(CircleGrads, self).__init__()

        if isinstance(kernel_sizes, int):
            assert isinstance(paddings, int), 'if kernel_sizes is in paddings should be int too'
            kernel_sizes = [kernel_sizes]
            paddings = [paddings]
        else:
            kernel_sizes = sorted(list(kernel_sizes))
            paddings = sorted(list(paddings))
            assert len(kernel_sizes) == len(paddings), 'there should be equal number of kernel_sizes and paddings'

        for i, kernel_size in enumerate(kernel_sizes):
            weight = get_circle_grads_kernel_nd(channels, kernel_size, dim)
            self.register_buffer('weight%d' % i, weight)

        self.groups = channels
        self.paddings = paddings
        self.dim = dim
        self.mode = mode

        if dim == 1:
            self.conv = F.conv1d
        elif dim == 2:
            self.conv = F.conv2d
        elif dim == 3:
            self.conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        if self.mode == 'rep_in':
            self.c_factor = len(kernel_sizes) * (1 + 1)
        elif self.mode == 'split_out':
            self.c_factor = (len(kernel_sizes) * 1) + 1
        else:
            raise RuntimeError('supported modes are rep_in and split_out')

    def forward(self, x):
        if self.mode == 'rep_in':
            g_out = []
        elif self.mode == 'split_out':
            g_out = [x]
        for i, padding in enumerate(self.paddings):
            if self.mode == 'rep_in':
                g_out.append(x)
            weight = getattr(self, 'weight%d' % i)
            g_out.append(self.conv(x, weight=weight, stride=1, padding=padding, groups=self.groups))
        return torch.cat(g_out, dim=1)


if __name__ == '__main__':
    import matplotlib.pyplot as plt
    import imageio
    import torchvision


    def close_event():
        plt.close()

    fig = plt.figure()
    timer = fig.canvas.new_timer(interval=500)
    timer.add_callback(close_event)

    c_size = 64
    # c_size = 128
    n_calls = 64
    # n_calls = 4

    canvas = torch.zeros([1, 1, c_size, c_size])
    # canvas[:, :, c_size // 2, c_size // 2] = 1.0
    # canvas[:, :, c_size // 2, c_size // 4] = 1.0

    plt.imshow(canvas.view(c_size, c_size))
    timer.start()
    plt.show()
    canvas_l = [canvas.view(c_size, c_size)]

    # kernel_sizes = [3, 31, 63]
    # kernel_sizes = [(2 ** i) + 1 for i in range(1, 5)]
    # kernel_sizes = [i + 1 for i in range(2, 10, 2)]
    kernel_sizes = [(2 ** i) + 1 for i in range(1, 7, 2)]
    # kernel_sizes = [3]
    kernels = []
    for i, kernel_size in enumerate(kernel_sizes):
        kernel = get_circle_grads_kernel_nd(1, kernel_size, 2)
        kernels.append(kernel)
    # paddings = [1, 15, 31]
    # paddings = [2 ** (i - 1) for i in range(1, 5)]
    # paddings = [i // 2 for i in range(2, 10, 2)]
    paddings = [2 ** (i - 1) for i in range(1, 7, 2)]
    # paddings = [1]
    # print(kernels[0].shape)
    for kernel in kernels:
        for i in range(kernel.shape[0]):
            plt.imshow(kernel[i, 0, ...].t())
            plt.show()
    print(kernel_sizes, paddings)

    for i in range(n_calls):
        canvas[:, :, c_size // 4, i] -= 1
        canvas[:, :, i, i] -= 1
        if i == 0:
            canvas[:, :, c_size * 3 // 4, c_size * 3 // 4] -= 1
        else:
            canvas[:, :, c_size * 3 // 4, c_size * 3 // 4] += 1
        canvas_circle = [canvas]
        for circle_grad_f, pad_f in zip(kernels, paddings):
            canvas_circle.append(F.conv2d(canvas, weight=circle_grad_f, stride=1, padding=pad_f))
        canvas = torch.cat(canvas_circle, dim=1)
        canvas = F.instance_norm(canvas)
        canvas = canvas.mean(dim=1, keepdim=True)
        canvas[:, :, c_size // 2:(c_size // 2) + 10, c_size // 2:(c_size // 2) + 10] = 0.0
        plt.imshow(canvas.view(c_size, c_size))
        timer.start()
        plt.show()
        canvas_l.append(canvas.view(c_size, c_size))

    timer.stop()
    plt.imshow(canvas.view(c_size, c_size))
    plt.show()

    def _canvas(idx):
        return canvas_l[idx]

    imageio.mimsave('./circle_grad_waves.gif', [_canvas(i) for i in range(n_calls)], fps=5)

    circle_grad_waves_steps = torch.stack(canvas_l[1:], dim=0).view(n_calls, 1, c_size, c_size)
    mi = circle_grad_waves_steps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    ma = circle_grad_waves_steps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    circle_grad_waves_steps = (circle_grad_waves_steps - mi) / (ma - mi)
    torchvision.utils.save_image(circle_grad_waves_steps, './circle_grad_waves_steps.png', nrow=8)
