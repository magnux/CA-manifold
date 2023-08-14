import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock


class DynaResidualU(nn.Module):
    def __init__(self, lat_size, fin, fout, image_size, n_layers, layer_factor=1, down_inits=True, down_gates=True, lat_factor=None):
        super().__init__()
        self.lat_size = lat_size
        self.fin = fin
        self.fout = fout
        self.image_size = image_size
        self.n_layers = n_layers
        self.down_gates = down_gates
        self.lat_factor = 1 if lat_factor is None else lat_factor

        self.f_size = (np.concatenate([np.linspace(fin, layer_factor * (fin + fout) / 2, n_layers), np.linspace(layer_factor * (fin + fout) / 2, fout, n_layers + 1)]) *
                       np.concatenate([np.linspace(1, n_layers, n_layers), np.linspace(n_layers, 1, n_layers + 1)])).astype(int).tolist()

        self.down_inits = None
        if down_inits:
            self.down_inits = nn.ParameterList([
                nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.f_size[l], 1, 1)).repeat(1, 1, self.image_size // 2 ** l, self.image_size // 2 ** l))
            for l in range(self.n_layers)])
        self.down_grads = nn.ModuleList([
            SinSobel(self.f_size[l] * (2 if down_inits else 1), 3, 1)
        for l in range(self.n_layers)])
        self.down_convs = nn.ModuleList([
            DynaResidualBlock(lat_size, self.f_size[l] * 3 * (2 if down_inits else 1), self.f_size[l + 1] * (2 if down_gates else 1), self.f_size[l], lat_factor=self.lat_factor)
        for l in range(self.n_layers)])

        self.up_grads = nn.ModuleList([
            SinSobel(self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0), 3, 1)
        for l in range(self.n_layers, self.n_layers * 2)])
        self.up_convs = nn.ModuleList([
            DynaResidualBlock(lat_size, (self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0)) * 3, self.f_size[l + 1], self.f_size[l + 1], lat_factor=self.lat_factor)
        for l in range(self.n_layers, self.n_layers * 2)])

    def forward(self, x, lat):
        x_l = []
        new_x = x
        for l in range(self.n_layers):
            if self.down_inits is not None:
                new_x = torch.cat([new_x, self.down_inits[l].repeat(x.shape[0], 1, 1, 1)], 1)
            new_x = self.down_grads[l](new_x)
            new_x = self.down_convs[l](new_x, lat)
            if self.down_gates:
                new_x, new_x_gate = torch.chunk(new_x, 2, dim=1)
                new_x = new_x * F.relu(new_x_gate + 1)
            if l < self.n_layers - 1:
                x_l.append(new_x)
                new_x = F.interpolate(new_x, scale_factor=0.5, mode='bilinear', align_corners=False)

        for l in range(self.n_layers):
            if l > 0:
                new_x = torch.cat([new_x, x_l[self.n_layers - 1 - l]], dim=1)
            new_x = self.up_grads[l](new_x)
            new_x = self.up_convs[l](new_x, lat)
            if l < self.n_layers - 1:
                new_x = F.interpolate(new_x, scale_factor=2., mode='bilinear', align_corners=False)

        return new_x
