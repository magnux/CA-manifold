import torch
from torch import nn
from torch.nn import functional as F


def get_rand_grads_kernel_nd(channels, kernel_size, dim):
    return torch.nn.init.orthogonal_(torch.empty([channels, 1]+[kernel_size for _ in range(dim)])) / channels ** 0.5


class RandGrads(nn.Module):
    def __init__(self, channels, kernel_sizes, paddings, dim=2, mode='split_out'):
        super(RandGrads, self).__init__()

        if isinstance(kernel_sizes, int):
            assert isinstance(paddings, int), 'if kernel_sizes is in paddings should be int too'
            kernel_sizes = [kernel_sizes]
            paddings = [paddings]
        else:
            kernel_sizes = sorted(list(kernel_sizes))
            paddings = sorted(list(paddings))
            assert len(kernel_sizes) == len(paddings), 'there should be equal number of kernel_sizes and paddings'

        for i, kernel_size in enumerate(kernel_sizes):
            weight = get_rand_grads_kernel_nd(channels, kernel_size, dim)
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

    canvas = torch.zeros([1, 3, c_size, c_size])
    # canvas[:, :, c_size // 2, c_size // 2] = 1.0
    # canvas[:, :, c_size // 2, c_size // 4] = 1.0

    plt.imshow(canvas.permute(0, 2, 3, 1).view(c_size, c_size, 3))
    timer.start()
    plt.show()
    canvas_l = [canvas.permute(0, 2, 3, 1).view(c_size, c_size, 3)]

    # kernel_sizes = [3, 31, 63]
    # kernel_sizes = [(2 ** i) + 1 for i in range(1, 5)]
    # kernel_sizes = [i + 1 for i in range(2, 10, 2)]
    kernel_sizes = [(2 ** i) + 1 for i in range(1, 7, 2)]
    # kernel_sizes = [3]
    kernels = []
    for i, kernel_size in enumerate(kernel_sizes):
        kernel = get_rand_grads_kernel_nd(3, kernel_size, 2)
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
        # canvas[:, :, c_size // 4, i] -= 1
        canvas[:, :, i, i] -= 1
        # if i == 0:
        #     canvas[:, :, c_size * 3 // 4, c_size * 3 // 4] -= 1
        # else:
        #     canvas[:, :, c_size * 3 // 4, c_size * 3 // 4] *= 1.1
        canvas_rand = [canvas]
        for rand_grad_f, pad_f in zip(kernels, paddings):
            canvas_rand.append(F.conv2d(canvas, weight=rand_grad_f, stride=1, padding=pad_f, groups=3))
        canvas = torch.cat(canvas_rand, dim=1)
        canvas = F.instance_norm(canvas)
        canvas = canvas.view(1, 4, 3, c_size, c_size).mean(dim=1)
        canvas[:, :, c_size // 2:(c_size // 2) + 10, c_size // 2:(c_size // 2) + 10] = 0.0
        plt.imshow(canvas.permute(0, 2, 3, 1).view(c_size, c_size, 3))
        timer.start()
        plt.show()
        canvas_l.append(canvas.permute(0, 2, 3, 1).view(c_size, c_size, 3))

    timer.stop()
    plt.imshow(canvas.permute(0, 2, 3, 1).view(c_size, c_size, 3))
    plt.show()

    def _canvas(idx):
        return canvas_l[idx]

    # imageio.mimsave('./rand_grad_waves.gif', [_canvas(i) for i in range(n_calls)], fps=5)
    #
    # rand_grad_waves_steps = torch.stack(canvas_l[1:], dim=0).view(n_calls, 3, c_size, c_size)
    # mi = rand_grad_waves_steps.min(dim=2, keepdim=True)[0].min(dim=3, keepdim=True)[0]
    # ma = rand_grad_waves_steps.max(dim=2, keepdim=True)[0].max(dim=3, keepdim=True)[0]
    # rand_grad_waves_steps = (rand_grad_waves_steps - mi) / (ma - mi)
    # torchvision.utils.save_image(rand_grad_waves_steps, './rand_grad_waves_steps.png', nrow=8)
