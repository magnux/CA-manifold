import torch
import torch.nn as nn
from src.layers.irm import IRMLinear


class NoiseInjection(nn.Module):
    def __init__(self, fin):
        super().__init__()
        self.fin = fin
        self.lat_to_fin = IRMLinear(fin, exp_mult=True)

    def forward(self, x, noise=None):
        batch_size = x.shape[0]

        squeeze_x = False
        if x.dim() == 2:
            x = x.unsqueeze(2)
            squeeze_x = True

        sp_size = [int(i) for i in x.shape[2:]]
        x_dim = x.dim() - 2

        if noise is None:
            noise = torch.randn([batch_size] + sp_size + [self.fin], device=x.device)
        noise = self.lat_to_fin(noise).permute(*[0, x_dim+1] + [i for i in range(1, x_dim+1)])

        x_new = x + noise

        if squeeze_x:
            x_new = x_new.squeeze(2)

        return x_new
