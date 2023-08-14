import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblocks import ResidualBlockS
from src.layers.linearresidualblocks import LinearResidualBlockS
from src.layers.sequentialcond import SequentialCond


class ResidualUS(nn.Module):
    def __init__(self, fin, fout, image_size, enc_position=True, n_condition=0, lat_size=0):
        super().__init__()
        self.fin = fin
        self.fout = fout
        self.image_size = image_size
        self.n_layers = int(np.log2(image_size)) - 1
        self.enc_position = enc_position
        self.n_condition = n_condition
        self.lat_size = lat_size

        self.f_size = (np.concatenate([np.linspace(fin, (fin + fout) / 2, self.n_layers), np.linspace((fin + fout) / 2, fout, self.n_layers + 1)]) *
                       np.concatenate([np.linspace(1, 2, self.n_layers), np.linspace(2, 1, self.n_layers + 1)])).astype(int).tolist()

        # FIXME: Remove the following lines before launching next train
        # self.cond_tmp_fc = LinearResidualBlockS(self.lat_size, int(self.lat_size ** 0.5) * self.n_layers)
        # self.cond_tmp_conv = nn.ModuleList([
        #     ResidualBlockS(self.f_size[l], self.f_size[l],
        #                    pos_enc=self.enc_position, image_size=self.image_size // 2 ** l,
        #                    condition=True, lat_size=int(self.lat_size ** 0.5))
        # for l in range(self.n_layers)])

        for d in range(self.n_condition):
            self.register_module('cond_fc_%d'%d, nn.Sequential(
                LinearResidualBlockS(self.lat_size, self.lat_size),
                LinearResidualBlockS(self.lat_size, int(self.lat_size ** 0.5) * self.n_layers)
            ))
            self.register_module('cond_conv_%d' % d, nn.ModuleList([
                SequentialCond(
                    ResidualBlockS(self.f_size[l], self.f_size[l],
                                   pos_enc=self.enc_position, image_size=self.image_size // 2 ** l,
                                   condition=True, lat_size=int(self.lat_size ** 0.5)),
                    ResidualBlockS(self.f_size[l], self.f_size[l],
                                   pos_enc=self.enc_position, image_size=self.image_size // 2 ** l,
                                   condition=True, lat_size=int(self.lat_size ** 0.5))
                )
            for l in range(self.n_layers)]))

        self.down_convs = nn.ModuleList([
            nn.Sequential(
                ResidualBlockS(self.f_size[l], self.f_size[l + 1],
                               pos_enc=self.enc_position, image_size=self.image_size // 2 ** l, attention=True, attn_patches=max(16, self.image_size // 2 ** (l + 2))),
                ResidualBlockS(self.f_size[l + 1], self.f_size[l + 1],
                               pos_enc=self.enc_position, image_size=self.image_size // 2 ** l, attention=True, attn_patches=max(16, self.image_size // 2 ** (l + 2))),
            )
        for l in range(self.n_layers)])

        self.up_convs = nn.ModuleList([
            nn.Sequential(
                ResidualBlockS(self.f_size[l] + (self.f_size[self.n_layers + (self.n_layers - l)] if l > self.n_layers else 0), self.f_size[l + 1],
                               pos_enc=self.enc_position, image_size=self.image_size // 2 ** ((self.n_layers * 2) - 1 - l), attention=True, attn_patches=max(16, self.image_size // 2 ** ((self.n_layers * 2) + 1 - l))),
                ResidualBlockS(self.f_size[l + 1], self.f_size[l + 1],
                               pos_enc=self.enc_position, image_size=self.image_size // 2 ** ((self.n_layers * 2) - 1 - l), attention=True, attn_patches=max(16, self.image_size // 2 ** ((self.n_layers * 2) + 1 - l))),
            )
        for l in range(self.n_layers, self.n_layers * 2)])

    def forward(self, x, lat=None):
        # assert (lat is None) == (not self.condition)

        cond_lat = []
        if not lat is None:
            for d in range(self.n_condition):
                if not lat[d] is None:
                    cond_lat_tmp = self.get_submodule('cond_fc_%d' % d)(lat[d])
                    cond_lat.append(torch.chunk(cond_lat_tmp, self.n_layers, 1))
                else:
                    cond_lat.append(None)

        x_l = []
        new_x = x
        for l in range(self.n_layers):
            if not lat is None:
                for d in range(self.n_condition):
                    if not cond_lat[d] is None:
                        new_x = self.get_submodule('cond_conv_%d' % d)[l](new_x, cond_lat[d][l])

            new_x = self.down_convs[l](new_x)

            if l < self.n_layers - 1:
                x_l.append(new_x)
                new_x = F.interpolate(new_x, scale_factor=0.5, mode='bilinear', align_corners=False)

        for l in range(self.n_layers):
            if l > 0:
                new_x = torch.cat([new_x, x_l[self.n_layers - 1 - l]], dim=1)

            new_x = self.up_convs[l](new_x)

            if l < self.n_layers - 1:
                new_x = F.interpolate(new_x, scale_factor=2., mode='bilinear', align_corners=False)

        return new_x
