import torch
import torch.nn as nn
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
import numpy as np


class LatentCube(nn.Module):
    def __init__(self, lat_size, n_filter, n_calls):
        super(LatentCube).__init__()
        self.lat_size = lat_size
        self.n_filter = n_filter
        self.n_calls = n_calls

        self.frac_sobel = SinSobel(self.n_filter, [3, 5], [1, 2], dim=3)
        self.frac_norm = nn.InstanceNorm3d(self.n_filter * self.frac_sobel.c_factor)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * self.frac_sobel.c_factor, self.n_filter, self.n_filter, dim=3)

        self.cube_size = int(np.ceil(self.lat_size ** (1 / 3)))
        self.pad_size = int((self.cube_size ** 3) - self.z_dim)
        self.register_buffer('cube_pad', torch.zeros(1, self.pad_size))

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter)).view(1, self.n_filter, 1, 1, 1).repeat(1, 1, self.cube_size, self.cube_size, self.cube_size))

    def forward(self, lat):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        out = torch.cat([self.seed.to(float_type)] * batch_size, 0)
        out = out.repeat(1, self.n_filter, 1, 1, 1)

        for c in range(self.n_calls):
            out_new = self.frac_sobel(out)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            out = out + (0.1 * out_new)

        out = out.mean(1).view(batch_size, -1)
        out = out[:, :self.lat_size]

        return out
