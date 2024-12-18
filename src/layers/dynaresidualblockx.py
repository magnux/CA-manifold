import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblockx import LinearResidualBlockX


class DynaResidualBlockX(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2, kernel_size=1, stride=1, padding=0, groups=1,
                 lat_factor=1):
        super(DynaResidualBlockX, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden
        self.dim = dim

        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.k_in_size = self.fhidden * (self.fin // groups) * (kernel_size ** dim)
        self.k_mida_size = self.fhidden * (self.fhidden // groups) * (kernel_size ** dim)
        self.k_midb_size = self.fhidden * (self.fhidden // groups) * (kernel_size ** dim)
        self.k_out_size = self.fout * (self.fhidden // groups) * (kernel_size ** dim)
        self.k_short_size = self.fout * (self.fin // groups) * (kernel_size ** dim)

        self.b_in_size = self.fhidden
        self.b_mid_size = self.fhidden
        self.b_out_size = self.fout
        self.b_short_size = self.fout

        k_total_size = (self.k_in_size + self.k_mida_size + self.k_midb_size +self.k_out_size + self.k_short_size +
                        self.b_in_size + self.b_mida_size + self.b_midb_size + self.b_out_size + self.b_short_size)
        if lat_factor == 0:
            self.dyna_k = nn.Linear(lat_size, k_total_size)
        else:
            self.dyna_k = LinearResidualBlockX(self.lat_size, k_total_size, int(self.lat_size * lat_factor) * 2)

        self.prev_lat = None
        self.k_in, self.k_mida, self.k_midb, self.k_out, self.k_short = None, None, None, None, None
        self.b_in, self.b_mida, self.b_midb, self.b_out, self.b_short = 0, 0, 0, 0, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k_in, k_mida, k_midb, k_out, k_short, b_in, b_mida, b_midb, b_out, b_short = torch.split(ks, [self.k_in_size, self.k_mida_size,
                                                                                                          self.k_midb_size, self.k_out_size,
                                                                                                          self.k_short_size,
                                                                                                          self.b_in_size, self.b_mida_size,
                                                                                                          self.b_midb_size, self.b_out_size,
                                                                                                          self.b_short_size], dim=1)

            self.k_in = k_in.view([batch_size, self.fhidden, self.fin // self.groups] + self.kernel_size) / self.fhidden ** 0.5
            self.k_mida = k_mida.view([batch_size, self.fhidden, self.fhidden // self.groups] + self.kernel_size) / self.fhidden ** 0.5
            self.k_midb = k_midb.view([batch_size, self.fhidden, self.fhidden // self.groups] + self.kernel_size) / self.fhidden ** 0.5
            self.k_out = k_out.view([batch_size, self.fout, self.fhidden // self.groups] + self.kernel_size) / self.fout ** 0.5
            self.k_short = k_short.view([batch_size, self.fout, self.fin // self.groups] + self.kernel_size) / self.fout ** 0.5

            self.k_in = self.k_in.reshape([batch_size * self.fhidden, self.fin // self.groups] + self.kernel_size)
            self.k_mida = self.k_mida.reshape([batch_size * self.fhidden, self.fhidden // self.groups] + self.kernel_size)
            self.k_midb = self.k_midb.reshape([batch_size * self.fhidden, self.fhidden // self.groups] + self.kernel_size)
            self.k_out = self.k_out.reshape([batch_size * self.fout, self.fhidden // self.groups] + self.kernel_size)
            self.k_short = self.k_short.reshape([batch_size * self.fout, self.fin // self.groups] + self.kernel_size)

            self.b_in = b_in.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
            self.b_mida = b_mida.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
            self.b_midb = b_midb.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
            self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            self.b_short = b_short.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])

            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new_s = self.f_conv(x_new, self.k_short, stride=self.stride, padding=self.padding, groups=batch_size * self.groups) + self.b_short
        x_new = self.f_conv(x_new, self.k_in, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_in
        x_new = F.relu(x_new, True)
        x_new = self.f_conv(x_new, self.k_mida, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_mida
        x_new = F.relu(x_new, True)
        x_new = self.f_conv(x_new, self.k_midb, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_midb
        x_new = F.relu(x_new, True)
        x_new = self.f_conv(x_new, self.k_out, stride=self.stride, padding=self.padding,
                            groups=batch_size * self.groups) + self.b_out
        x_new = x_new + x_new_s
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
