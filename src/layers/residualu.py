import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.sobel import SinSobel
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.posencoding import sin_cos_pos_encoding_nd


class ResidualU(nn.Module):
    def __init__(self, fin, fout, image_size, layer_factor=4, enc_position=True, auto_condition=True, ext_condition=False, lat_size=512):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.image_size = image_size
        self.n_layers = int(np.log2(image_size)) + 1
        self.enc_position = enc_position
        self.auto_condition = auto_condition
        self.ext_condition = ext_condition
        self.lat_size = lat_size

        self.f_size = (np.concatenate([np.linspace(fin, layer_factor * (fin + fout) / 2, self.n_layers), np.linspace(layer_factor * (fin + fout) / 2, fout, self.n_layers + 1)]) *
                       np.concatenate([np.linspace(1, self.n_layers, self.n_layers), np.linspace(self.n_layers, 1, self.n_layers + 1)])).astype(int).tolist()

        if self.enc_position:
            for l in range(self.n_layers - 1):
                self.register_buffer('pos_enc_%d' % l, sin_cos_pos_encoding_nd(self.image_size // 2 ** l, 2, 1))

        self.down_grads = nn.ModuleList([
            SinSobel(self.f_size[l], 3, 1)
        for l in range(self.n_layers)])
        self.down_convs = nn.ModuleList([
            ResidualBlock(self.f_size[l] * 3 + (getattr(self, 'pos_enc_%d' % l).size(1) if enc_position and l < self.n_layers - 1 else 0), self.f_size[l + 1], self.f_size[l], 1, 1, 0)
        for l in range(self.n_layers)])

        self.up_grads = nn.ModuleList([
            SinSobel(self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0), 3, 1)
        for l in range(self.n_layers, self.n_layers * 2)])
        self.up_convs = nn.ModuleList([
            ResidualBlock((self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0)) * 3, self.f_size[l + 1], self.f_size[l + 1], 1, 1, 0)
        for l in range(self.n_layers, self.n_layers * 2)])

        if self.auto_condition or self.ext_condition:
            if self.auto_condition:
                self.auto_lat = LinearResidualBlock(self.f_size[self.n_layers], self.lat_size)

            self.cond = nn.ModuleList([
                nn.Linear(self.lat_size * (2 if self.auto_condition and self.ext_condition else 1), (self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0)) * 3 * 2)
            for l in range(self.n_layers, self.n_layers * 2)])

    def forward(self, x, lat=None):
        assert (lat is None) == (not self.ext_condition)
        x_l = []
        new_x = x
        for l in range(self.n_layers):
            new_x = self.down_grads[l](new_x)

            if self.enc_position and l < self.n_layers - 1:
                pos_enc = getattr(self, 'pos_enc_%d' % l)
                pos_enc = pos_enc.repeat(x.shape[0], 1, 1, 1)
                new_x = torch.cat([new_x, pos_enc], 1)

            new_x = self.down_convs[l](new_x)
            if l < self.n_layers - 1:
                x_l.append(new_x)
                new_x = F.interpolate(new_x, scale_factor=0.5, mode='bilinear', align_corners=False)

        if self.auto_condition:
            auto_lat = self.auto_lat(new_x.view(x.size(0), -1))
            if lat is not None:
                lat = torch.cat([lat, auto_lat], 1)
            else:
                lat = auto_lat

        lat = F.gelu(lat)
        for l in range(self.n_layers):
            if l > 0:
                new_x = torch.cat([new_x, x_l[self.n_layers - 1 - l]], dim=1)
            new_x = self.up_grads[l](new_x)

            if self.auto_condition or self.ext_condition:
                cond = self.cond[l](lat).view(new_x.size(0), new_x.size(1) * 2, 1, 1)
                cond_m, cond_s = torch.chunk(cond, 2, 1)
                new_x = new_x * cond_s + cond_m

            new_x = self.up_convs[l](new_x)
            if l < self.n_layers - 1:
                new_x = F.interpolate(new_x, scale_factor=2., mode='bilinear', align_corners=False)

        return new_x
