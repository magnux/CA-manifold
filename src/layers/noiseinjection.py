import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.equallinear import EqualLinear


class NoiseInjection(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.fin = fin
        self.noise_weights = nn.Parameter(torch.zeros(1, self.fin))

    def forward(self, x, noise=None):
        if noise is None:
            noise = torch.randn_like(x)

        x_new = x + self.noise_weights.expand_as(x) * noise

        return x_new
