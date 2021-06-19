import torch
import torch.nn as nn
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.expmult import ExpMult


class NoiseInjection(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.lat_to_fin = nn.Sequential(
            nn.Linear(1, fin),
            ExpMult(fin),
            LinearResidualBlock(fin, fin),
            LinearResidualBlock(fin, fin),
        )

    def forward(self, x, noise=None):
        batch_size = x.size(0)

        squeeze_x = False
        if x.dim() == 2:
            x = x.unsqueeze(2)
            squeeze_x = True

        in_size = x.size(2)
        x_dim = x.dim() - 2

        if noise is None:
            noise = torch.randn([batch_size] + [in_size for _ in range(x_dim)] + [1], device=x.device)
        noise = self.lat_to_fin(noise).permute(*[0, x_dim+1] + [i for i in range(2, x_dim+1)] + [1])

        x_new = x + noise

        if squeeze_x:
            x_new = x_new.squeeze(2)

        return x_new
