import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class MixConv(nn.Module):
    def __init__(self, lat_size, fin, fout, dim=2, kernel_size=1, stride=1, padding=0, groups=1, lat_factor=1, n_mix=8):
        super(MixConv, self).__init__()

        self.lat_size = lat_size if lat_size > 3 else 512
        self.fin = fin
        self.fout = fout
        self.dim = dim
        self.n_mix = n_mix

        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.kernel_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout, self.fin // groups] + [kernel_size for _ in range(dim)]))))
        self.bias_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout] + [1 for _ in range(dim)]))))
        
        if lat_factor == 0:
            self.dyna_mix = nn.Linear(lat_size, n_mix)
        else:
            self.dyna_mix = nn.Sequential(
                nn.Linear(lat_size, self.lat_size * lat_factor),
                LinearResidualBlock(self.lat_size * lat_factor, self.lat_size * lat_factor),
                LinearResidualBlock(self.lat_size * lat_factor, n_mix),
            )

        self.prev_lat = None
        self.kernel, self.bias = None, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x, lat):
        batch_size = x.size(0)
        
        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_mix(lat)

            ks = ks.view([batch_size, self.n_mix, 1, 1] + [1 for _ in range(self.dim)])
            self.kernel = (self.kernel_mix * ks).sum(1).reshape([batch_size * self.fout, self.fin // self.groups] + self.kernel_size)

            ks = ks.view([batch_size, self.n_mix, 1] + [1 for _ in range(self.dim)])
            self.bias = (self.bias_mix * ks).sum(1).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new = self.f_conv(x_new, self.kernel, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.bias
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
