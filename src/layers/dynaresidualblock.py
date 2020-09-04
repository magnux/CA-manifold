import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.residualblock import ResidualBlock
from src.layers.residualattentionblock import ResidualAttentionBlock
from src.layers.lambd import LambdaLayer
from itertools import chain
from math import ceil


class DynaResidualBlock(nn.Module):
    def __init__(self, lat_size, fin, fout, fhidden=None, dim=2, kernel_size=1, padding=0, norm_weights=False):
        super(DynaResidualBlock, self).__init__()

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

        self.k_in_size = self.fhidden * self.fin * (kernel_size ** dim)
        self.k_mid_size = self.fhidden * self.fhidden * (kernel_size ** dim)
        self.k_out_size = self.fout * self.fhidden * (kernel_size ** dim)
        self.k_short_size = self.fout * self.fin * (kernel_size ** dim)
        
        self.b_in_size = self.fhidden if not norm_weights else 0
        self.b_mid_size = self.fhidden if not norm_weights else 0
        self.b_out_size = self.fout if not norm_weights else 0
        self.b_short_size = self.fout if not norm_weights else 0

        k_total_size = (self.k_in_size + self.k_mid_size + self.k_out_size + self.k_short_size +
                        self.b_in_size + self.b_mid_size + self.b_out_size + self.b_short_size)

        n_blocks = 2
        self.dyna_k = nn.Sequential(
            nn.Linear(self.lat_size, ceil(k_total_size / self.fhidden) * self.fhidden),
            LambdaLayer(lambda x: x.view(x.size(0), self.fhidden, ceil(k_total_size / self.fhidden))),
            * list(chain(*[[ResidualAttentionBlock(ceil(k_total_size / self.fhidden), self.fhidden, self.fhidden // 4),
                            ResidualBlock(self.fhidden, self.fhidden, None, 1, 1, 0, nn.Conv1d)] for _ in range(n_blocks)])),
            LambdaLayer(lambda x: x.view(x.size(0), ceil(k_total_size / self.fhidden) * self.fhidden)),
            nn.Linear(ceil(k_total_size / self.fhidden) * self.fhidden, k_total_size)
        )

        self.prev_lat = None
        self.k_in, self.k_mid, self.k_out, self.k_short = None, None, None, None
        self.b_in, self.b_mid, self.b_out, self.b_short = 0, 0, 0, 0
        self.kernel_size = [kernel_size for _ in range(self.dim)]
        self.padding = padding
        self.norm_weights = norm_weights

    def forward(self, x, lat):
        batch_size = x.size(0)
        
        if self.prev_lat is None or self.prev_lat.data_ptr() != lat.data_ptr():
            ks = self.dyna_k(lat)
            k_in, k_mid, k_out, k_short, b_in, b_mid, b_out, b_short = torch.split(ks, [self.k_in_size, self.k_mid_size,
                                                                                        self.k_out_size, self.k_short_size,
                                                                                        self.b_in_size, self.b_mid_size,
                                                                                        self.b_out_size, self.b_short_size], dim=1)
            self.k_in = k_in.view([batch_size, self.fhidden, self.fin] + self.kernel_size)
            self.k_mid = k_mid.view([batch_size, self.fhidden, self.fhidden] + self.kernel_size)
            self.k_out = k_out.view([batch_size, self.fout, self.fhidden] + self.kernel_size)
            self.k_short = k_short.view([batch_size, self.fout, self.fin] + self.kernel_size)

            if self.norm_weights:
                self.k_in = self.k_in / (self.k_in + 1e-4).norm(dim=2, keepdim=True)
                self.k_mid = self.k_mid / (self.k_mid + 1e-4).norm(dim=2, keepdim=True)
                self.k_out = self.k_out / (self.k_out + 1e-4).norm(dim=2, keepdim=True)
                self.k_short = self.k_short / (self.k_short + 1e-4).norm(dim=2, keepdim=True)

            self.k_in = self.k_in.reshape([batch_size * self.fhidden, self.fin] + self.kernel_size)
            self.k_mid = self.k_mid.reshape([batch_size * self.fhidden, self.fhidden] + self.kernel_size)
            self.k_out = self.k_out.reshape([batch_size * self.fout, self.fhidden] + self.kernel_size)
            self.k_short = self.k_short.reshape([batch_size * self.fout, self.fin] + self.kernel_size)

            if not self.norm_weights:
                self.b_in = b_in.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_mid = b_mid.view([batch_size, self.fhidden]).reshape([1, batch_size * self.fhidden] + [1 for _ in range(self.dim)])
                self.b_out = b_out.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
                self.b_short = b_short.view([batch_size, self.fout]).reshape([1, batch_size * self.fout] + [1 for _ in range(self.dim)])
            
            self.prev_lat = lat

        x_new = x.reshape([1, batch_size * self.fin] + [x.size(d + 2) for d in range(self.dim)])
        x_new_s = self.f_conv(x_new, self.k_short, groups=batch_size, padding=self.padding) + self.b_short
        x_new = self.f_conv(x_new, self.k_in, groups=batch_size, padding=self.padding) + self.b_in
        x_new = F.leaky_relu(x_new)
        x_new = self.f_conv(x_new, self.k_mid, groups=batch_size, padding=self.padding) + self.b_mid
        x_new = F.leaky_relu(x_new)
        x_new = self.f_conv(x_new, self.k_out, groups=batch_size, padding=self.padding) + self.b_out
        x_new = x_new + x_new_s
        x_new = x_new.reshape([batch_size, self.fout] + [x.size(d + 2) for d in range(self.dim)])
        
        return x_new
