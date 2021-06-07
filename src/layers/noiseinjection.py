import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.equallinear import EqualLinear


class NoiseInjection(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.lat_to_fin = nn.Linear(1, fin)

    def forward(self, x, noise=None):
        batch_size = x.size(0)
        in_size = x.size(2)
        x_dim = x.dim() - 2

        if noise is None:
            noise = torch.randn([batch_size] + [in_size for _ in range(x_dim)] + [1], device=x.device)
        noise = self.lat_to_fin(noise).permute(*[0, x_dim+1] + [i for i in range(2, x_dim+1)] + [1])

        x_new = x + noise

        return x_new
