import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblocks import ResidualBlockS
from src.layers.linearresidualblocks import LinearResidualBlockS
from src.layers.residualus import ResidualUS
from src.layers.posencoding import PosEncoding
from src.layers.sequentialcond import SequentialCond
from src.layers.irm import IRMConv
from src.networks.base import LabsEncoder
import numpy as np


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, z_out=False, z_dim=0, zoom_factor=0, norm_out=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.n_calls = n_calls
        self.lat_size = lat_size
        self.zoom_factor = zoom_factor
        self.norm_out = norm_out

        self.n_layers = int(np.log2(image_size)) - 1 + zoom_factor

        self.conv_img = ResidualBlockS(self.in_chan, self.n_filter)

        self.cond_fc = nn.Sequential(
            LinearResidualBlockS(self.lat_size, self.lat_size),
            LinearResidualBlockS(self.lat_size, int(self.lat_size ** 0.5) * self.n_layers)
        )

        self.layer_conv = nn.ModuleList([
            SequentialCond(
                ResidualBlockS(self.n_filter * int((l / self.n_layers) + 1), self.n_filter * int(((l + 1) / self.n_layers) + 1), pos_enc=True, image_size=(2 ** self.zoom_factor * self.image_size) // 2 ** l, condition=True, lat_size=int(self.lat_size ** 0.5)),
                ResidualBlockS(self.n_filter * int(((l + 1) / self.n_layers) + 1), self.n_filter * int(((l + 1) / self.n_layers) + 1), pos_enc=True, image_size=(2 ** self.zoom_factor * self.image_size) // 2 ** l, condition=True, lat_size=int(self.lat_size ** 0.5)),
            )
        for l in range(self.n_layers)])

        out_size = self.lat_size if not z_out else z_dim
        self.out_to_lat = nn.Sequential(
            LinearResidualBlockS(self.n_filter * 2 * 16, out_size),
            LinearResidualBlockS(out_size, out_size)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'

        cond_lat = self.cond_fc(inj_lat)
        cond_lat = torch.chunk(cond_lat, self.n_layers, 1)

        out = x
        if self.zoom_factor > 0:
            out = F.interpolate(out, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
        out = self.conv_img(out)

        out_embs = [out]

        for l in range(self.n_layers):
            out = self.layer_conv[l](out, cond_lat[l])
            if l < self.n_layers - 1:
                out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

            out_embs.append(out)

        lat = self.out_to_lat(out.flatten(1))
        if self.norm_out:
            lat = F.normalize(lat)

        return lat, out_embs, None


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        self.inj_lat = self.labs_encoder(labels)
        # if g_factor > 0.:
        #     self.inj_lat = (1. - g_factor) * self.inj_lat + g_factor * torch.randn_like(self.inj_lat)
        return super().forward(x, self.inj_lat)

    @property
    def lr_mul(self):
        return self.labs_encoder.yembed_to_lat.lr_mul

    @lr_mul.setter
    def lr_mul(self, lr_mul):
        self.labs_encoder.yembed_to_lat.lr_mul = lr_mul


class ZInjectedEncoder(LabsInjectedEncoder):
    def __init__(self, **kwargs):
        kwargs['z_out'] = True
        super().__init__(**kwargs)


class NInjectedEncoder(LabsInjectedEncoder):
    def __init__(self, **kwargs):
        kwargs['norm_out'] = True
        super().__init__(**kwargs)


class Decoder(nn.Module):
    def __init__(self, lat_size, image_size, channels, n_filter, n_conds=1,
                 zoom_factor=0, gravity=False, reversible=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.image_size = image_size
        self.n_filter = n_filter
        self.zoom_factor = zoom_factor
        self.reversible = reversible

        self.lat_size = lat_size

        self.gravity = gravity

        self.in_conv = IRMConv(self.out_chan, self.n_filter, int(np.log2(image_size)) - 1)

        # self.labs_encoder = LabsEncoder(lat_size=lat_size, **kwargs)

        self.rev_swich_emb = nn.Linear(1, self.lat_size)

        self.cell_to_cell = ResidualUS(self.n_filter, self.n_filter * (2 if gravity else 1), (2 ** self.zoom_factor * self.image_size), True, n_conds + 1 if reversible else n_conds, self.lat_size)

        self.out_conv = ResidualBlockS(self.n_filter, self.out_chan)

    def forward(self, img_init, conds, rev_switch=None):

        if isinstance(conds, torch.Tensor):
            conds = (conds,)

        cell_in = img_init
        if self.zoom_factor > 0:
            cell_in = F.interpolate(cell_in, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
        cell_in = self.in_conv(cell_in)

        rev_switch_emb = None
        if self.reversible:
            if rev_switch is not None:
                rev_switch_emb = self.rev_swich_emb(rev_switch)

        cell_out = self.cell_to_cell(cell_in, conds + (rev_switch_emb,) if self.reversible is not None else conds)

        if self.gravity:
            cell_out, cell_out_g = torch.chunk(cell_out, 2, 1)
            # cell_out = cell_out * (torch.relu(cell_out_g + 1) + 1e-4)
            cell_out = cell_out * torch.exp(cell_out_g)

        if self.zoom_factor > 0:
            cell_out = F.interpolate(cell_out, size=self.image_size, mode='bilinear', align_corners=False)

        return self.out_conv(cell_out)
