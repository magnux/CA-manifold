import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.equallinear import EqualLinear
import numpy as np


class ModConv(nn.Module):
    def __init__(self, lat_size, fin, fout, kernel, stride=1, dilation=1, dim=2, n_layers_style=4, **kwargs):
        super().__init__()
        
        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))
        self.dim = dim

        self.fin = fin
        self.fout = fout
        self.kernel = kernel
        self.stride = stride
        self.dilation = dilation
        weight = torch.randn([fout, fin] + [kernel for _ in range(dim)])
        nn.init.kaiming_normal_(weight, a=0, mode='fan_in', nonlinearity='leaky_relu')
        self.weight = nn.Parameter(weight)

        lat_to_styles = []
        for _ in range(n_layers_style):
            lat_to_styles.extend([EqualLinear(lat_size, lat_size), nn.LeakyReLU(0.2, True)])
        lat_to_styles.append(nn.Linear(lat_size, fin))
        self.lat_to_styles = nn.Sequential(*lat_to_styles)
        self.prev_lat = None
        self.mod_weight = None

    def _get_same_padding(self, size, kernel, dilation, stride):
        return ((size - 1) * (stride - 1) + dilation * (kernel - 1)) // 2

    def forward(self, x, lat):
        batch_size = x.size(0)
        in_size = x.size(2)
        padding = self._get_same_padding(in_size, self.kernel, self.dilation, self.stride)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            self.prev_lat = lat
            styles = self.lat_to_styles(lat)

            mod_weight = self.weight * (1 / np.sqrt(self.fin * self.kernel ** 2) / self.weight.norm(float('inf'), dim=[i for i in range(1, self.dim + 2)], keepdim=True)) # max_Ikk
            styles = styles / styles.norm(float('inf'), dim=1, keepdim=True)  # max_I

            mod_weight = mod_weight.unsqueeze(0)
            mod_weight = mod_weight * styles.reshape([batch_size, 1, self.fin] + [1 for _ in range(self.dim)])

            dcoefs = (mod_weight.square().sum(dim=[i for i in range(2, self.dim + 3)]) + 1e-8).rsqrt() # [NO]
            mod_weight = mod_weight * dcoefs.reshape(batch_size, -1, 1, 1, 1) # [NOIkk]

            _, _, *ws = mod_weight.shape
            mod_weight = mod_weight.reshape([batch_size * self.fout, self.fin] + [self.kernel for _ in range(self.dim)])

            self.mod_weight = mod_weight

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new = self.f_conv(x_new, mod_weight, padding=padding, groups=batch_size)
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
