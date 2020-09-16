import torch
import torch.nn as nn


class NormScaleAndShift(nn.Module):

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        s = s.clamp(-1e-3, 1.)  # Safety first
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        return z

    def backward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        s = s.clamp(-1e-3, 1.)  # Safety first
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        return x


class DeNormScaleAndShift(nn.Module):

    def __init__(self, dim, scale=True, shift=True):
        super().__init__()
        self.s = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if scale else None
        self.t = nn.Parameter(torch.randn(1, dim, requires_grad=True)) if shift else None

    def forward(self, z):
        s = self.s if self.s is not None else z.new_zeros(z.size())
        s = s.clamp(-1e-3, 1.)  # Safety first
        t = self.t if self.t is not None else z.new_zeros(z.size())
        x = (z - t) * torch.exp(-s)
        return x

    def backward(self, x):
        s = self.s if self.s is not None else x.new_zeros(x.size())
        s = s.clamp(-1e-3, 1.)  # Safety first
        t = self.t if self.t is not None else x.new_zeros(x.size())
        z = x * torch.exp(s) + t
        return z
