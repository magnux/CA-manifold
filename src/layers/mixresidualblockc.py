import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblockc import LinearResidualBlockC
from src.layers.complexwave import complex_wave


class MixResidualBlockC(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2, kernel_size=1, stride=1, padding=0, groups=1, lat_factor=1, n_mix=8):
        super(MixResidualBlockC, self).__init__()

        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden
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

        self.k_in_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, (self.fhidden // 2), self.fin // groups] + [kernel_size for _ in range(dim)]))))
        self.k_mid_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, (self.fhidden // 2), 2 * (self.fhidden // 2) // groups] + [kernel_size for _ in range(dim)]))))
        self.k_out_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout, 2 * (self.fhidden // 2) // groups] + [kernel_size for _ in range(dim)]))))
        self.k_short_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout, self.fin // groups] + [kernel_size for _ in range(dim)]))))
        
        self.b_in_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, (self.fhidden // 2)] + [1 for _ in range(dim)]))))
        self.b_mid_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, (self.fhidden // 2)] + [1 for _ in range(dim)]))))
        self.b_out_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout] + [1 for _ in range(dim)]))))
        self.b_short_mix = nn.Parameter(torch.nn.init.xavier_normal_(torch.empty(*([1, n_mix, self.fout] + [1 for _ in range(dim)]))))

        if lat_factor == 0:
            self.dyna_mix = nn.Linear(lat_size, n_mix)
        else:
            self.dyna_mix = nn.Sequential(
                LinearResidualBlockC(self.lat_size, int(self.lat_size * lat_factor)),
                LinearResidualBlockC(int(self.lat_size * lat_factor), n_mix),
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
            ks = self.dyna_mix(lat)

            ks = ks.view([batch_size, self.n_mix, 1, 1] + [1 for _ in range(self.dim)])
            self.k_in = (self.k_in_mix * ks).sum(1).reshape([batch_size * (self.fhidden // 2), self.fin // self.groups] + self.kernel_size)
            self.k_mid = (self.k_mid_mix * ks).sum(1).reshape([batch_size * (self.fhidden // 2), 2 * (self.fhidden // 2) // self.groups] + self.kernel_size)
            self.k_out = (self.k_out_mix * ks).sum(1).reshape([batch_size * self.fout, 2 * (self.fhidden // 2) // self.groups] + self.kernel_size)
            self.k_short = (self.k_short_mix * ks).sum(1).reshape([batch_size * self.fout, self.fin // self.groups] + self.kernel_size)

            ks = ks.view([batch_size, self.n_mix, 1] + [1 for _ in range(self.dim)])
            self.b_in = (self.b_in_mix * ks).sum(1).reshape([1, batch_size * (self.fhidden // 2)] + [1 for _ in range(self.dim)])
            self.b_mid = (self.b_mid_mix * ks).sum(1).reshape([1, batch_size * (self.fhidden // 2)] + [1 for _ in range(self.dim)])
            self.b_out = (self.b_out_mix * ks).sum(1).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            self.b_short = (self.b_short_mix * ks).sum(1).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
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
