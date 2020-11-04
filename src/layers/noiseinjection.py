import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.equallinear import EqualLinear


class NoiseInjection(nn.Module):
    def __init__(self, lat_size, fin, dim=2, n_layers_style=4):
        super().__init__()
        self.dim = dim

        lat_to_y = []
        for _ in range(n_layers_style):
            lat_to_y.extend([EqualLinear(lat_size, lat_size), nn.LeakyReLU(0.2, True)])
        lat_to_y.append(nn.Linear(lat_size, fin))
        self.lat_to_fin = nn.Sequential(*lat_to_y)

    def forward(self, x, noise=None):
        batch_size = x.size(0)
        in_size = x.size(2)

        if noise is None:
            noise = torch.FloatTensor(*([batch_size] + [in_size for _ in range(self.dim)] + [1])).uniform_(0., 1.).cuda(x.device)
        noise = self.lat_to_fin(noise).transpose(self.dim + 1, 1)

        x_new = x + noise

        return x_new
