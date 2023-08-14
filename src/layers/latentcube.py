import torch
import torch.nn as nn
from src.layers.gaussgrads import GaussGrads
from src.layers.dynaresidualblock import DynaResidualBlock
import numpy as np


class LatentCube(nn.Module):
    def __init__(self, lat_size, out_size, n_filter, n_calls):
        super(LatentCube, self).__init__()

        self.lat_size = lat_size
        self.out_size = out_size
        self.cube_size = int(np.ceil(self.out_size ** (1 / 3)))
        self.n_filter = n_filter
        self.n_calls = n_calls

        self.frac_gauss = GaussGrads(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(self.cube_size)-1), 1)],
                                                    [2 ** (i - 1) for i in range(1, int(np.log2(self.cube_size) - 1), 1)], dim=3,  mode='rep_in')
        self.frac_factor = self.frac_gauss.c_factor
        self.frac_groups = self.frac_gauss.c_factor // 5
        self.frac_norm = nn.InstanceNorm3d(self.n_filter * self.frac_factor)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * self.frac_factor,
                                                self.n_filter * self.frac_groups, self.n_filter * self.frac_groups,
                                                dim=3)

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter)).view(1, self.n_filter, 1, 1, 1).repeat(1, 1, self.cube_size, self.cube_size, self.cube_size))

    def forward(self, lat, out=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if out is None:
            out = torch.cat([self.seed.to(float_type)] * batch_size, 0)

        for _ in range(self.n_calls):
            out_new = self.frac_gauss(out)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            out_new = out_new.reshape(batch_size, self.n_filter, self.frac_groups, self.cube_size, self.cube_size, self.cube_size).sum(dim=2)
            out = out + 0.1 * out_new

        lat = out.mean(1).view(batch_size, -1)
        lat = lat[:, :self.out_size]

        return lat, out
