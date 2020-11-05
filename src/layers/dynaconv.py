import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaConv(nn.Module):
    def __init__(self, lat_size, fin, fout, kernel_size=1, stride=1, padding=0, dim=2, norm_weights=False):
        super(DynaConv, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.dim = dim

        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.k_size = self.fout * self.fin * (kernel_size ** dim)
        self.b_size = self.fout if not norm_weights else 0

        self.dyna_k = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.k_size + self.b_size, self.lat_size * 2),
        )

        self.prev_lat = None
        self.k, self.b = None, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.norm_weights = norm_weights

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k, b = torch.split(ks, [self.k_size, self.b_size], dim=1)
            self.k = k.view([batch_size, self.fout, self.fin] + self.kernel_size)
            if self.norm_weights:
                self.k = self.k * torch.rsqrt((self.k ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
            self.k = self.k.reshape([batch_size * self.fout, self.fin] + self.kernel_size)

            if not self.norm_weights:
                self.b = b.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])

            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new = self.f_conv(x_new, self.k, stride=self.stride, padding=self.padding, groups=batch_size) + self.b
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
