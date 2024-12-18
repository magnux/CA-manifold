import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblockc import LinearResidualBlockC
from src.layers.complexwave import complex_wave


class DynaResidualBlockC(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, stride=1, kernel_size=1, padding=0, groups=1, dim=2, lat_factor=1):
        super(DynaResidualBlockC, self).__init__()

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

        self.k_in_size = (self.fhidden // 2) * (self.fin // groups) * (kernel_size ** dim)
        self.k_mid_size = (self.fhidden // 2) * (2 * (self.fhidden // 2) // groups) * (kernel_size ** dim)
        self.k_out_size = self.fout * (2 * (self.fhidden // 2) // groups) * (kernel_size ** dim)
        self.k_short_size = self.fout * (self.fin // groups) * (kernel_size ** dim)
        
        self.b_in_size = (self.fhidden // 2)
        self.b_mid_size = (self.fhidden // 2)
        self.b_out_size = self.fout
        self.b_short_size = self.fout

        k_total_size = (self.k_in_size + self.k_mid_size + self.k_out_size + self.k_short_size +
                        self.b_in_size + self.b_mid_size + self.b_out_size + self.b_short_size)
        if lat_factor == 0:
            self.dyna_k = nn.Linear(lat_size, k_total_size)
        else:
            self.dyna_k = nn.Sequential(
                LinearResidualBlockC(self.lat_size, int(self.lat_size * lat_factor)),
                LinearResidualBlockC(int(self.lat_size * lat_factor), k_total_size, int(self.lat_size * lat_factor) * 2),
            )

        self.prev_lat = None
        self.k_in, self.k_mid, self.k_out, self.k_short = None, None, None, None
        self.b_in, self.b_mid, self.b_out, self.b_short = 0, 0, 0, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.groups = groups

    def forward(self, x, lat):
        batch_size = x.size(0)

        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k_in, k_mid, k_out, k_short, b_in, b_mid, b_out, b_short = torch.split(ks, [self.k_in_size, self.k_mid_size,
                                                                                        self.k_out_size, self.k_short_size,
                                                                                        self.b_in_size, self.b_mid_size,
                                                                                        self.b_out_size, self.b_short_size], dim=1)

            self.k_in = k_in.view([batch_size, (self.fhidden // 2), self.fin // self.groups] + self.kernel_size) / self.fhidden ** 0.5
            self.k_mid = k_mid.view([batch_size, (self.fhidden // 2), 2 * (self.fhidden // 2) // self.groups] + self.kernel_size) / self.fhidden ** 0.5
            self.k_out = k_out.view([batch_size, self.fout, 2 * (self.fhidden // 2) // self.groups] + self.kernel_size) / self.fout ** 0.5
            self.k_short = k_short.view([batch_size, self.fout, self.fin // self.groups] + self.kernel_size) / self.fout ** 0.5

            self.k_in = self.k_in.reshape([batch_size * (self.fhidden // 2), self.fin // self.groups] + self.kernel_size)
            self.k_mid = self.k_mid.reshape([batch_size * (self.fhidden // 2), 2 * (self.fhidden // 2) // self.groups] + self.kernel_size)
            self.k_out = self.k_out.reshape([batch_size * self.fout, 2 * (self.fhidden // 2) // self.groups] + self.kernel_size)
            self.k_short = self.k_short.reshape([batch_size * self.fout, self.fin // self.groups] + self.kernel_size)

            self.b_in = b_in.view([batch_size, (self.fhidden // 2)]).reshape([1, batch_size * (self.fhidden // 2)] + [1 for _ in range(self.dim)])
            self.b_mid = b_mid.view([batch_size, (self.fhidden // 2)]).reshape([1, batch_size * (self.fhidden // 2)] + [1 for _ in range(self.dim)])
            self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            self.b_short = b_short.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new_s = self.f_conv(x_new, self.k_short, stride=self.stride, padding=self.padding, groups=batch_size * self.groups) + self.b_short
        x_new = self.f_conv(x_new, self.k_in, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_in
        x_new = complex_wave(x_new)
        x_new = self.f_conv(x_new, self.k_mid, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_mid
        x_new = complex_wave(x_new)
        x_new = self.f_conv(x_new, self.k_out, stride=self.stride, padding=self.padding, groups=batch_size * self.groups) + self.b_out
        x_new = x_new + x_new_s
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])

        return x_new
