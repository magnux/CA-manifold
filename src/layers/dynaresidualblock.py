import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock


class DynaResidualBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2, kernel_size=1, stride=1, padding=0, groups=1, lat_factor=1, norm_weights=False, complexify=False):
        super(DynaResidualBlock, self).__init__()
        assert not complexify or ((fin % 2 == 0) and (fout % 2 == 0)), 'complexify only works with an even number of channels'

        self.lat_size = lat_size if lat_size > 3 else 512
        self.fin = fin
        self.fout = fout
        self.fhidden = max((fin + fout), 1) if fhidden is None else fhidden
        self.dim = dim
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.stride = stride
        self.padding = padding
        self.norm_weights = norm_weights
        self.groups = groups * (2 if complexify else 1)
        self.complexify = complexify

        if dim == 1:
            self.f_conv = F.conv1d
        elif dim == 2:
            self.f_conv = F.conv2d
        elif dim == 3:
            self.f_conv = F.conv3d
        else:
            raise RuntimeError('Only 1, 2 and 3 dimensions are supported. Received {}.'.format(dim))

        self.k_in_size = self.fhidden * (self.fin // self.groups) * (kernel_size ** dim)
        self.k_mid_size = self.fhidden * (self.fhidden // self.groups) * (kernel_size ** dim)
        self.k_out_size = self.fout * (self.fhidden // self.groups) * (kernel_size ** dim)
        self.k_short_size = self.fout * (self.fin // self.groups) * (kernel_size ** dim)
        
        self.b_in_size = self.fhidden if not norm_weights else 0
        self.b_mid_size = self.fhidden if not norm_weights else 0
        self.b_out_size = self.fout if not norm_weights else 0
        self.b_short_size = self.fout if not norm_weights else 0

        k_total_size = (self.k_in_size + self.k_mid_size + self.k_out_size + self.k_short_size +
                        self.b_in_size + self.b_mid_size + self.b_out_size + self.b_short_size)

        self.dyna_k = nn.Sequential(
            nn.Linear(lat_size, self.lat_size * lat_factor),
            LinearResidualBlock(self.lat_size * lat_factor, self.lat_size * lat_factor),
            LinearResidualBlock(self.lat_size * lat_factor, k_total_size, self.lat_size * lat_factor * 2),
        )

        self.prev_lat = None
        self.k_in, self.k_mid, self.k_out, self.k_short = None, None, None, None
        self.b_in, self.b_mid, self.b_out, self.b_short = 0, 0, 0, 0

        if self.complexify:
            group_size = self.fin // self.groups
            self.real_idx = torch.cat([torch.arange(group_size) + (i * group_size) for i in range(0, self.groups, 2)])
            self.imaginary_idx = self.real_idx + group_size
            self.complex_idx = torch.cat([torch.cat([torch.arange(group_size, group_size * 2), torch.arange(group_size)]) + (i * group_size) for i in range(0, self.groups, 2)])

    def forward(self, x, lat):
        batch_size = x.size(0)
        
        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k_in, k_mid, k_out, k_short, b_in, b_mid, b_out, b_short = torch.split(ks, [self.k_in_size, self.k_mid_size,
                                                                                        self.k_out_size, self.k_short_size,
                                                                                        self.b_in_size, self.b_mid_size,
                                                                                        self.b_out_size, self.b_short_size], dim=1)

            self.k_in = k_in.view([batch_size, self.fhidden, self.fin // self.groups] + self.kernel_size)
            self.k_mid = k_mid.view([batch_size, self.fhidden, self.fhidden // self.groups] + self.kernel_size)
            self.k_out = k_out.view([batch_size, self.fout, self.fhidden // self.groups] + self.kernel_size)
            self.k_short = k_short.view([batch_size, self.fout, self.fin // self.groups] + self.kernel_size)

            if self.norm_weights:
                self.k_in = self.k_in * torch.rsqrt((self.k_in ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_mid = self.k_mid * torch.rsqrt((self.k_mid ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_out = self.k_out * torch.rsqrt((self.k_out ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)
                self.k_short = self.k_short * torch.rsqrt((self.k_short ** 2).sum(dim=[i for i in range(2, self.dim + 3)], keepdim=True) + 1e-8)

            self.k_in = self.k_in.reshape([batch_size * self.fhidden, self.fin // self.groups] + self.kernel_size)
            self.k_mid = self.k_mid.reshape([batch_size * self.fhidden, self.fhidden // self.groups] + self.kernel_size)
            self.k_out = self.k_out.reshape([batch_size * self.fout, self.fhidden // self.groups] + self.kernel_size)
            self.k_short = self.k_short.reshape([batch_size * self.fout, self.fin // self.groups] + self.kernel_size)

            if not self.norm_weights:
                self.b_in = b_in.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_mid = b_mid.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
                self.b_short = b_short.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
            self.prev_lat = lat

        def _dyna_conv(x_new):
            x_new_s = self.f_conv(x_new, self.k_short, stride=self.stride, padding=self.padding, groups=batch_size * self.groups) + self.b_short
            x_new = self.f_conv(x_new, self.k_in, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_in
            x_new = F.relu(x_new, True)
            x_new = self.f_conv(x_new, self.k_mid, stride=1, padding=self.padding, groups=batch_size * self.groups) + self.b_mid
            x_new = F.relu(x_new, True)
            x_new = self.f_conv(x_new, self.k_out, stride=self.stride, padding=self.padding, groups=batch_size * self.groups) + self.b_out
            x_new = x_new + x_new_s
            x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])
            return x_new

        if self.complexify:
            x_new = torch.empty_like(x)

            x_ri = x
            x_new_ri = x_ri.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
            x_ir = x[:, self.complex_idx, ...]
            x_new_ir = x_ir.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])

            x_new_ri = _dyna_conv(x_new_ri)
            x_new_ir = _dyna_conv(x_new_ir)

            x_new_prr = x_new_ri[:, self.real_idx, ...]
            x_new_pii = x_new_ri[:, self.imaginary_idx, ...]

            x_new[:, self.real_idx, ...] = x_new_prr - x_new_pii

            x_new_pir = x_new_ir[:, self.real_idx, ...]
            x_new_pri = x_new_ir[:, self.imaginary_idx, ...]

            x_new[:, self.imaginary_idx, ...] = x_new_pir + x_new_pri
        else:
            x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
            x_new = _dyna_conv(x_new)

        return x_new
