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
                ResidualBlockS(self.n_filter * (l + 1), self.n_filter * (l + 2), pos_enc=True, image_size=(2 ** self.zoom_factor * self.image_size) // 2 ** l, condition=True, lat_size=int(self.lat_size ** 0.5)),
                ResidualBlockS(self.n_filter * (l + 2), self.n_filter * (l + 2), pos_enc=True, image_size=(2 ** self.zoom_factor * self.image_size) // 2 ** l, condition=True, lat_size=int(self.lat_size ** 0.5)),
            )
        for l in range(self.n_layers)])

        out_size = self.lat_size if not z_out else z_dim
        self.out_to_lat = nn.Sequential(
            LinearResidualBlockS(self.n_filter * (self.n_layers + 1) * 16, out_size),
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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, gated=True, env_feedback=False, multi_cut=True, auto_reg=False, ce_out=False, zoom_factor=0, letter_encoding=False, letter_channels=2, letter_bits=8, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        if letter_encoding:
            self.lat_size = lat_size * letter_channels * letter_bits
        else:
            self.lat_size = lat_size
        self.n_calls = n_calls
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.ce_out = ce_out
        self.zoom_factor = zoom_factor

        self.in_conv = ResidualBlockS(self.out_chan, self.n_filter)

        self.call_enc = PosEncoding(int(self.lat_size ** 0.5), 0, 4, self.n_calls)

        self.cell_to_cell = ResidualUS(self.n_filter, self.n_filter, (2 ** self.zoom_factor * self.image_size), True, True, self.lat_size + self.call_enc.size())

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.image_size + (1 if self.causal else 0), self.image_size + (1 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

        if self.log_mix_out:
            out_f = 10 * ((self.out_chan * 3) + 1)
        elif self.ce_out:
            out_f = self.out_chan * 256
            ce_pos = torch.arange(0, 256).view(1, 256, 1, 1, 1)
            ce_pos = ce_pos.expand(-1, -1, self.out_chan, self.image_size, self.image_size)
            self.register_buffer('ce_pos', ce_pos)
        else:
            out_f = self.out_chan

        self.out_conv = ResidualBlockS(self.n_filter, out_f)

    def forward(self, lat, img_init, call_idcs):

        out = img_init

        out_embs = [out]
        if self.causal:
            out = F.pad(out, [0, 1, 0, 1])

        cell_in = out

        if self.zoom_factor > 0:
            cell_in = F.interpolate(cell_in, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
        cell_in = self.in_conv(cell_in)

        if lat is not None:
            call_enc = self.call_enc(lat, call_idcs)
        else:
            call_enc = None
        cell_out = self.cell_to_cell(cell_in, call_enc)

        if self.zoom_factor > 0:
            cell_out = F.interpolate(cell_out, size=self.image_size, mode='bilinear', align_corners=False)

        out_embs.append(cell_out)

        noise_out = self.out_conv(cell_out)

        return noise_out, out_embs, None
