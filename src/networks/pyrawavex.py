import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.sobel import SinSobel
from src.layers.mixresidualblock import MixResidualBlock
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.gaussiansmoothing import GaussianSmoothing
from src.layers.hard_nl import hardsigmoid
from src.layers.lambd import LambdaLayer
from src.layers.expscale import ExpScale
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain

# from src.networks.conv_ae import Encoder


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, z_out=False, z_dim=0, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_layers = int(np.log2(image_size)) - 1

        self.conv_img = nn.Conv2d(self.in_chan, self.n_filter, 3, 1, 1)
        self.frac_sobel = nn.ModuleList([SinSobel(self.n_filter * (l + 1), 3, 1) for l in range(self.n_layers)])
        self.frac_conv = nn.ModuleList([MixResidualBlock(self.lat_size, self.n_filter * (l + 1) * 3, self.n_filter * (l + 2), self.n_filter * (l + 1)) for l in range(self.n_layers)])
        self.frac_ds = nn.ModuleList([nn.Sequential(
            GaussianSmoothing(self.n_filter * (l + 2), 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)),
        ) for l in range(self.n_layers)])

        self.out_to_lat = LinearResidualBlock(self.n_filter * (self.n_layers + 1) * 4, self.lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = self.conv_img(x)

        out_embs = [out]
        out_embs_raw = []
        for l in range(self.n_layers):
            out = self.frac_sobel[l](out)
            out_embs_raw.append(out)
            out = F.instance_norm(out)

            out = self.frac_conv[l](out, inj_lat)
            out = self.frac_ds[l](out)

            out_embs.append(out)

        out_embs_raw.append(out)
        lat = self.out_to_lat(out.reshape(batch_size, self.n_filter * (self.n_layers + 1) * 4))

        return lat, out_embs, out_embs_raw


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


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate, log_mix_out=False, gated=True, ce_out=False, n_seed=1, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.n_calls = n_calls
        self.lat_size = lat_size
        self.n_layers = int(np.log2(image_size)) - 1
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.log_mix_out = log_mix_out
        self.gated = gated
        self.ce_out = ce_out
        self.n_seed = n_seed

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.image_size // (2 ** (self.n_layers -1)), self.image_size // (2 ** (self.n_layers -1))))

        self.wave_inits = nn.ParameterList([
            nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter * (l + 1), 1, 1)).repeat(1, 1, self.image_size // (2 ** l), self.image_size // (2 ** l)))
        for l in range(self.n_layers - 1, -1, -1)])

        self.wave_grads = nn.ModuleList([
            SinSobel(self.n_filter * (l + 1), 3, 1)
        for l in range(self.n_layers - 1, -1, -1)])

        self.wave_convs = nn.ModuleList([
            DynaResidualBlock(self.lat_size, self.n_filter * (l + 1) * 3, self.n_filter, self.n_filter * (l + 1), lat_factor=(self.lat_size ** 0.9) / self.lat_size)
        for l in range(self.n_layers - 1, -1, -1)])

        self.frac_sobel = SinSobel(self.n_filter, 3, 1)

        self.frac_convs = nn.ModuleList([
            DynaResidualBlock(self.lat_size, self.n_filter * (3 + 1), self.n_filter * (2 if self.gated else 1), self.n_filter * 2 * (2 if self.gated else 1))
        for l in range(self.n_layers - 1, -1, -1)])

        self.lat_exp_factors = nn.ModuleList([ExpScale(self.lat_size) for l in range(self.n_layers * self.n_calls)])
        self.leak_factors = nn.ParameterList([nn.Parameter(torch.ones([]) * 0.1) for l in range(self.n_layers)])

        self.gauss_smooth = GaussianSmoothing(self.n_filter, 3, 1, 1)

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

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, 16, lat.device).to(float_type)
            if isinstance(seed_n, tuple):
                seed = self.seed[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                seed = self.seed[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                seed = self.seed[seed_n:seed_n + 1, ...]
            out = seed.to(float_type).repeat(batch_size, 1, 1, 1)
        else:
            out = ca_init

        out_embs = [out]
        for l in range(0 if ca_init is None else self.n_layers - 1, self.n_layers):
            wave = self.wave_inits[l]

            wave = self.wave_grads[l](wave)
            wave = wave.repeat(batch_size, 1, 1, 1)

            exp_lat = self.lat_exp_factors[l](lat)
            wave = self.wave_convs[l](wave, exp_lat)
            wave = F.instance_norm(wave)

            leak_factor = torch.clamp(self.leak_factors[l], 1e-3, 1e3)
            for c in range(self.n_calls):
                out_new = self.frac_sobel(out)
                out_new = F.instance_norm(out_new)

                out_new = torch.cat([out_new, wave], dim=1)
                out_new = self.frac_convs[l](out_new, exp_lat)

                if self.gated:
                    out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                    out_new = out_new * hardsigmoid(out_new_gate)

                out = out + (leak_factor * out_new)

            if l < self.n_layers -1:
                out = F.interpolate(out, scale_factor=2, mode='bilinear', align_corners=False)
                out = self.gauss_smooth(out)

            out_embs.append(out)

        if not float(np.log2(self.image_size)).is_integer():
            out = F.interpolate(out, size=self.image_size, mode='bilinear', align_corners=False)

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
