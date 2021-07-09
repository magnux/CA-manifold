import torch
from torch import nn
from torch.nn import functional as F
import numpy as np


def get_gauss_grads_kernel_nd(channels, kernel_size, dim, n_diff, left_sided=False):
    filter_space = np.linspace(-1., 1., kernel_size)
    if n_diff == 1:
        gauss_grads = -1/4 * np.exp(-2 * filter_space ** 2) * filter_space
    elif n_diff == 2:
        gauss_grads = 1/4 * np.exp(-2 * filter_space ** 2) * (-1 + 4 * filter_space ** 2)
    else:
        raise RuntimeError(
            'Only 1 and 2 diffs are implemented'.format(dim)
        )
    if left_sided:
        gauss_grads[int(kernel_size / 2):] = 0
    gauss_grads = torch.tensor(gauss_grads, dtype=torch.float32).view(*([1, 1, kernel_size] + [1 for _ in range(dim - 1)]))
    if dim > 1:
        gauss_grads = gauss_grads.repeat(*([1, 1, 1] + [kernel_size for _ in range(dim - 1)]))
        gauss_grads_l = [gauss_grads]
        for i in range(3, dim + 2):
            gauss_grads_l.append(gauss_grads.transpose(2, i))
        gauss_grads = torch.cat(gauss_grads_l, dim=0)

    gauss_grads_kernel = gauss_grads.repeat(*([channels, 1] + [1 for _ in range(dim)]))

    return gauss_grads_kernel


class GaussGrads(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings, dim=2, left_sided=False, rep_in=False):
        super(GaussGrads, self).__init__()

        if isinstance(kernel_sizes, int):
            assert isinstance(paddings, int), 'if kernel_sizes is in paddings should be int too'
            kernel_sizes = [kernel_sizes]
            paddings = [paddings]
        else:
            kernel_sizes = sorted(list(kernel_sizes))
            paddings = sorted(list(paddings))
            assert len(kernel_sizes) == len(paddings), 'there should be equal number of kernel_sizes and paddings'

        for i, kernel_size in enumerate(kernel_sizes):
            for d in [1, 2]:
                self.register_buffer('weight%d%d' % (i, d), get_gauss_grads_kernel_nd(channels, kernel_size, dim, d, left_sided))

        self.groups = channels
        self.paddings = paddings
        self.dim = dim
        self.rep_in = rep_in

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
        self.c_factor = len(kernel_sizes) * (dim * 2 + 1) if self.rep_in else (len(kernel_sizes) * dim * 2) + 1

    def forward(self, x):
        if self.rep_in:
            s_out = []
            for i, padding in enumerate(self.paddings):
                for d in [1, 2]:
                    weight = getattr(self, 'weight%d%d' % (i, d))
                    s_out.extend([x, self.conv(x, weight=weight, stride=1, padding=padding, groups=self.groups)])
            return torch.cat(s_out, dim=1)
        else:
            s_out = [x]
            for i, padding in enumerate(self.paddings):
                for d in [1, 2]:
                    weight = getattr(self, 'weight%d%d' % (i, d))
                    s_out.append(self.conv(x, weight=weight, stride=1, padding=padding, groups=self.groups))
            return torch.cat(s_out, dim=1)


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
    canvas[:, :, c_size // 2, c_size // 2] = 1.0
    canvas[:, :, c_size // 2, c_size // 4] = 1.0

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
        for d in [1, 2]:
            kernels.append(get_gauss_grads_kernel_nd(1, kernel_size, 2, d))
    kernel_sizes = np.repeat(kernel_sizes, 2)
    # paddings = [1, 15, 31]
    # paddings = [2 ** (i - 1) for i in range(1, 5)]
    # paddings = [i // 2 for i in range(2, 10, 2)]
    paddings = [2 ** (i - 1) for i in range(1, 7, 2)]
    paddings = np.repeat(paddings, 2)
    # paddings = [1]
    # print(kernels[0].shape)
    for i in range(kernels[0].shape[0]):
        plt.imshow(kernels[0][i, 0, ...].t())
        plt.show()
    print(kernel_sizes, paddings)

    for _ in range(n_calls):
        canvas_gauss = [canvas]
        for gauss_grad_f, pad_f in zip(kernels, paddings):
            canvas_gauss.append(F.conv2d(canvas, weight=gauss_grad_f, stride=1, padding=pad_f))
        canvas = torch.cat(canvas_gauss, dim=1)
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

    imageio.mimsave('./gauss_grad_waves.gif', [_canvas(i) for i in range(n_calls)], fps=5)

    gauss_grad_waves_steps = torch.stack(canvas_l[1:], dim=0).view(n_calls, 1, c_size, c_size)
    mi = gauss_grad_waves_steps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    ma = gauss_grad_waves_steps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    gauss_grad_waves_steps = (gauss_grad_waves_steps - mi) / (ma - mi)
    torchvision.utils.save_image(gauss_grad_waves_steps, './gauss_grad_waves_steps.png', nrow=8)
