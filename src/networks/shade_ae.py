import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.noiseinjection import NoiseInjection
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain

from src.networks.conv_ae import Encoder


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, multi_cut=True, z_out=False, z_dim=0, ce_in=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = n_calls
        self.multi_cut = multi_cut
        self.ce_in = ce_in

        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, self.n_filter, 1, 1, 0)

        self.frac_sobel = SinSobel(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(image_size)-1), 1)],
                                                  [2 ** (i - 1) for i in range(1, int(np.log2(image_size)-1), 1)], mode='split_out')

        self.frac_factor = self.frac_sobel.c_factor
        self.frac_groups = self.frac_sobel.c_factor // 3

        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * self.frac_factor, self.n_filter * self.frac_factor, self.n_filter * self.frac_factor, groups=self.frac_groups, lat_factor=2)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * self.frac_factor)
        self.frac_conv = ResidualBlock(self.n_filter * self.frac_factor, self.n_filter, None, 1, 1, 0)

        self.frac_lat = nn.ModuleList([LinearResidualBlock(self.lat_size, self.lat_size) for _ in range(self.n_calls)])

        self.out_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
        self.out_to_lat = nn.Linear(sum(self.conv_state_size), lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        out = self.in_conv(x)

        out_embs = [out]
        for c in range(self.n_calls):
            inj_lat = inj_lat + 0.1 * self.frac_lat[c](inj_lat)
            out_new = self.frac_sobel(out)
            out_new = self.frac_norm(out_new)
            out_new = out_new * torch.sigmoid(self.frac_dyna_conv(out_new, inj_lat))
            out_new = self.frac_conv(out_new)
            out = out + 0.1 * out_new
            out_embs.append(out)

        out = self.frac_conv(out)
        out = self.out_conv(out)
        if self.multi_cut:
            conv_state_f, conv_state_fh, conv_state_fw, conv_state_hw = torch.split(out, self.split_sizes, dim=1)
            conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
                                    conv_state_fh.mean(dim=3).view(batch_size, -1),
                                    conv_state_fw.mean(dim=2).view(batch_size, -1),
                                    conv_state_hw.view(batch_size, -1)], dim=1)
        else:
            conv_state = out.mean(dim=(2, 3))
        lat = self.out_to_lat(conv_state)

        return lat, out_embs, None


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        inj_lat = self.labs_encoder(labels)
        return super().forward(x, inj_lat)


class ZInjectedEncoder(LabsInjectedEncoder):
    def __init__(self, **kwargs):
        kwargs['z_out'] = True
        super().__init__(**kwargs)


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, log_mix_out=False, ce_out=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.log_mix_out = log_mix_out
        self.ce_out = ce_out

        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter)).reshape(1, self.n_filter, 1, 1))

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.image_size, self.image_size))

        self.frac_sobel = SinSobel(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(image_size)-1), 1)],
                                                  [2 ** (i - 1) for i in range(1, int(np.log2(image_size)-1), 1)], mode='split_out')

        self.frac_factor = self.frac_sobel.c_factor
        self.frac_groups = self.frac_sobel.c_factor // 3

        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * self.frac_factor, self.n_filter * self.frac_factor, self.n_filter * self.frac_factor, groups=self.frac_groups, lat_factor=2)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * self.frac_factor)
        self.frac_conv = ResidualBlock(self.n_filter * self.frac_factor, self.n_filter, None, 1, 1, 0)

        self.frac_lat = nn.ModuleList([LinearResidualBlock(self.lat_size, self.lat_size) for _ in range(self.n_calls)])

        self.frac_noise = nn.ModuleList([NoiseInjection(self.n_filter) for _ in range(self.n_calls)])

        if self.log_mix_out:
            out_f = 10 * ((self.out_chan * 3) + 1)
        elif self.ce_out:
            out_f = self.out_chan * 256
            ce_pos = torch.arange(0, 256).view(1, 256, 1, 1, 1)
            ce_pos = ce_pos.expand(-1, -1, self.out_chan, self.image_size, self.image_size)
            self.register_buffer('ce_pos', ce_pos)
        else:
            out_f = self.out_chan
        self.out_conv = nn.Conv2d(self.n_filter, out_f, 1, 1, 0)

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device).to(float_type)
            out = torch.cat([self.seed.to(float_type)] * batch_size, 0)
        else:
            proj = torch.cat([self.in_proj.to(float_type)] * batch_size, 0)
            out = ca_init + proj

        out_embs = [out]
        for c in range(self.n_calls):
            lat = lat + 0.1 * self.frac_lat[c](lat)
            out_new = self.frac_sobel(out)
            out_new = self.frac_norm(out_new)
            out_new = out_new * torch.sigmoid(self.frac_dyna_conv(out_new, lat))
            out_new = self.frac_conv(out_new)
            out_new = self.frac_noise[c](out_new)
            out = out + 0.1 * out_new
            out_embs.append(out)

        out = self.frac_conv(out)
        out = self.out_conv(out)
        if self.ce_out:
            out = out.view(batch_size, 256, self.out_chan, self.image_size, self.image_size)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        elif self.ce_out:
            # Differentiable
            pos = self.ce_pos.expand(batch_size, -1, -1, -1, -1)
            out_d = ((out.softmax(dim=1) * pos).sum(dim=1) / 127.5) - 1
            out_raw = (out_d, out_raw)
            # Non-Differentiable
            out = (out.argmax(dim=1) / 127.5) - 1
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw
