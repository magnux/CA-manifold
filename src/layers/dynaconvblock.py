import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaConvBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2):
        super(DynaConvBlock, self).__init__()
        # Attributes
        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout) * 2, 1) if fhidden is None else fhidden
        self.dim = dim
        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.k_in_size = self.fhidden * self.fin
        self.k_out_size = self.fout * self.fhidden

        self.b_in_size = self.fhidden
        self.b_out_size = self.fout

        self.dyna_k = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.k_in_size + self.k_out_size +
                                               self.b_in_size + self.b_out_size, self.lat_size * 2),
        )

        self.prev_lat = None
        self.k_in, self.k_out = None, None
        self.b_in, self.b_out = None, None
        self.kernel_size = [1 for _ in range(self.dim)]

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k_in, k_out, b_in, b_out = torch.split(ks, [self.k_in_size, self.k_out_size, self.b_in_size, self.b_out_size], dim=1)
            self.k_in = k_in.view([batch_size, self.fhidden, self.fin] + self.kernel_size).reshape([batch_size * self.fhidden, self.fin] + self.kernel_size)
            self.k_out = k_out.view([batch_size, self.fout, self.fhidden] + self.kernel_size).reshape([batch_size * self.fout, self.fhidden] + self.kernel_size)
            self.b_in = b_in.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
            self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])

            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new = self.f_conv(x_new, self.k_in, groups=batch_size) + self.b_in
        x_new = F.relu(x_new)
        x_new = self.f_conv(x_new, self.k_out, groups=batch_size) + self.b_out
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
