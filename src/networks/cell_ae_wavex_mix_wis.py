import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.sobel import SinSobel
from src.layers.randgrads import DynaRandGrads
from src.layers.mixresidualblock import MixResidualBlock
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.gaussiansmoothing import GaussianSmoothing
from src.layers.hard_nl import hardsigmoid
from src.layers.complexwave import complex_wave
from src.layers.lambd import LambdaLayer
from src.networks.base import LabsEncoder
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_layers = int(np.log2(image_size)) - 1
        self.auto_reg = auto_reg

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
        lat_embs = [out.mean(dim=(2, 3))]
        # lat = torch.zeros(batch_size, self.lat_size, device=x.device)
        for l in range(self.n_layers):
            out = self.frac_sobel[l](out)
            if not self.auto_reg:
                out = F.instance_norm(out)
            out = self.frac_conv[l](out, inj_lat)
            out = self.frac_ds[l](out)

            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
            out_embs.append(out)
            lat_embs.append(out.mean(dim=(2, 3)))

        lat = self.out_to_lat(out.reshape(batch_size, self.n_filter * (self.n_layers + 1) * 4))

        return lat, out_embs, lat_embs


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


class MultiLabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = nn.ModuleList(
            [LabsEncoder(**kwargs) for _ in range(self.lat_mult)]
        )

    def forward(self, x, labels):
        inj_lat = []
        for l in range(self.lat_mult):
            inj_lat.append(self.labs_encoder[l](labels))
        self.inj_lat = torch.cat(inj_lat, 1)
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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, gated=True, env_feedback=False, multi_cut=True, auto_reg=False, ce_out=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
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
        self.n_layers = int(np.log2(image_size)) - 1

        self.cell_grads = DynaRandGrads(self.lat_size, self.n_filter, [3, 5], [1, 2], 2, self.n_calls)

        self.wave_inits = nn.ParameterList([
            nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter * l, 1, 1)).repeat(1, 1, self.image_size // 2 ** (l - 1), self.image_size // 2 ** (l - 1)))
        for l in range(self.n_layers + 1, 0, -1)])
        self.wave_grads = nn.ModuleList([
            DynaRandGrads(self.lat_size, self.n_filter * l * (2 if l < self.n_layers + 1 else 1), 3, 1, lat_factor=(self.lat_size ** 0.9) / self.lat_size)
        for l in range(self.n_layers + 1, 0, -1)])
        self.wave_convs = nn.ModuleList([
            DynaResidualBlock(self.lat_size, self.n_filter * l * (2 if l < self.n_layers + 1 else 1) * 2, self.n_filter * max(l - 1, 1), self.n_filter * l, lat_factor=(self.lat_size ** 0.9) / self.lat_size)
        for l in range(self.n_layers + 1, 0, -1)])
        self.cell_to_cell = DynaResidualBlock(self.lat_size, self.n_filter * (self.cell_grads.c_factor + 2), self.n_filter * (2 if self.gated else 1), self.n_filter * 2)

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

        self.out_conv = nn.Conv2d(self.n_filter, out_f, 3, 1, 1)

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            out = torch.zeros(batch_size, self.n_filter, self.image_size, self.image_size, device=lat.device).to(float_type)
        else:
            out = ca_init

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        waves = []
        for l, (wave_init, wave_grad, wave_conv) in enumerate(zip(self.wave_inits, self.wave_grads, self.wave_convs)):
            wave_init = wave_init.repeat(batch_size, 1, 1, 1)
            if l > 0:
                wave = torch.cat([wave, wave_init], 1)
            else:
                wave = wave_init
            wave = wave_grad(wave, lat)
            wave = F.instance_norm(wave)
            wave = wave_conv(wave, lat)
            if l < self.n_layers:
                wave = F.interpolate(wave, scale_factor=2, mode='bilinear', align_corners=False)
            elif wave.shape[2] != self.image_size:
                wave = F.interpolate(wave, size=self.image_size, mode='bilinear', align_corners=False)
            waves.append(wave)

        wave = complex_wave(wave)

        out_embs = [out]
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))

            cell_grads = self.cell_grads(out_new, lat)
            cell_grads = F.instance_norm(cell_grads)

            out_new = torch.cat([cell_grads, wave], dim=1)

            out_new = self.cell_to_cell(out_new, lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * hardsigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=lat.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=lat.device).to(float_type))
            out = out + (0.1 * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
            out_embs.append(out)

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

        return out, out_embs, waves
