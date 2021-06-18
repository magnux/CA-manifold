import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.equallinear import EqualLinear


class NoiseInjection(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.lat_to_fin_u = nn.Linear(1, fin)
        self.lat_to_fin_n = nn.Linear(1, fin)

    def forward(self, x, noise=None):
        batch_size = x.size(0)

        squeeze_x = False
        if x.dim() == 2:
            x = x.unsqueeze(2)
            squeeze_x = True

        in_size = x.size(2)
        x_dim = x.dim() - 2

        if noise is None:
            noise_u = torch.rand([batch_size] + [in_size for _ in range(x_dim)] + [1], device=x.device)
            noise_n = torch.randn([batch_size] + [in_size for _ in range(x_dim)] + [1], device=x.device)
        else:
            noise_u, noise_n = noise

        noise_u = self.lat_to_fin_u(noise_u).permute(*[0, x_dim + 1] + [i for i in range(2, x_dim + 1)] + [1])
        noise_n = self.lat_to_fin_n(noise_n).permute(*[0, x_dim + 1] + [i for i in range(2, x_dim+1)] + [1])

        x_new = x + noise_u + noise_n

        if squeeze_x:
            x_new = x_new.squeeze(2)

        return x_new
