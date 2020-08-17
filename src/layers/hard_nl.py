import torch
import torch.nn as nn


def hardsigmoid(x, inplace=False):
    return nn.functional.relu6(x + 3, inplace=inplace) / 6


class HardSigmoid(nn.Module):
    def __init__(self, inplace=False):
        super(HardSigmoid, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hardsigmoid(x, self.inplace)


def hardswish(x, inplace=False):
    return x * hardsigmoid(x, inplace)


class HardSwish(nn.Module):
    def __init__(self, inplace=False):
        super(HardSwish, self).__init__()
        self.inplace = inplace

    def forward(self, x):
        return hardswish(x, self.inplace)


class MOHSigmoid(nn.Module):
    def __init__(self, dim, n_sigmoid, inplace=False):
        super(MOHSigmoid, self).__init__()
        self.dim = dim
        self.n_sigmoid = n_sigmoid
        self.inplace = inplace

    def forward(self, x):
        x = torch.split(x, x.size(self.dim) // self.n_sigmoid, dim=self.dim)
        x = torch.stack(x, dim=-1)
        x = hardsigmoid(x, self.inplace)
        x = x.mean(-1)
        return x
