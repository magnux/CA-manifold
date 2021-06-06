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
            noise_shape = [int(s) for s in x.shape]
            if x.dim() > 2:
                noise_shape[1] = 1
            noise = torch.randn(noise_shape, device=x.device)

        exp_weights = self.noise_weights
        if x.dim() > 2:
            exp_weights = exp_weights.view(*[1, self.fin] + [1 for _ in x.shape[2:]])
        x_new = x + exp_weights * noise

        return x_new
