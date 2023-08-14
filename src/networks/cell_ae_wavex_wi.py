import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.sobel import SinSobel
from src.layers.dynaresidualu import DynaResidualU
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.hard_nl import hardsigmoid
from src.layers.complexwave import complex_wave
from src.layers.lambd import LambdaLayer
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain

# from src.networks.conv_ae import Encoder


class SlimEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, causal=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, ce_in=False, embed_size=0, wavify_lat=False, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.causal = causal
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.ce_in = ce_in
        self.wavify_lat = wavify_lat

        self.labs_encoder = LabsEncoder(n_labels, lat_size, embed_size, auto_reg, **kwargs)

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        self.split_sizes = [self.n_filter] * 7 if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.image_size, self.image_size, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.n_filter]

        self.conv_to_lat = nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, sum(self.split_sizes), 3, 1, 1)
        self.lat_to_lat = nn.Linear(sum(self.conv_state_size), self.lat_size)
        self.lat_mix = nn.Linear(self.lat_size * 2, self.lat_size, self.lat_size * 2)

    def forward(self, x, labels):
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        labs_lat = self.labs_encoder(labels)

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        conv_state = self.conv_to_lat(x)
        if self.multi_cut:
            conv_state_f, conv_state_h, conv_state_w, conv_state_fh, conv_state_fw, conv_state_hw, conv_state_g = torch.split(conv_state, self.split_sizes, dim=1)
            conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
                                    conv_state_h.mean(dim=(1, 3)),
                                    conv_state_w.mean(dim=(1, 2)),
                                    conv_state_fh.mean(dim=3).view(batch_size, -1),
                                    conv_state_fw.mean(dim=2).view(batch_size, -1),
                                    conv_state_hw.mean(dim=1).view(batch_size, -1),
                                    conv_state_g.mean(dim=(1, 2, 3)).view(batch_size, 1)], dim=1)
        else:
            conv_state = conv_state.mean(dim=(2, 3))
        img_lat = self.lat_to_lat(conv_state)

        lat = self.lat_mix(torch.cat([img_lat, labs_lat], dim=1))

        if self.wavify_lat:
            lat = complex_wave(lat)

        return lat, None, None


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, causal=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, ce_in=False, wavify_lat=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.causal = causal
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.ce_in = ce_in
        self.n_layers = int(np.log2(image_size)) - 2
        self.wavify_lat = wavify_lat

        self.split_sizes = [self.n_filter] * 7 if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.image_size, self.image_size, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.n_filter]

        self.conv_img = nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, self.n_filter, 3, 1, 1)
        self.s_enc = SlimEncoder(n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate, skip_fire, causal, env_feedback, multi_cut, z_out, z_dim, auto_reg, ce_in, wavify_lat=wavify_lat, **kwargs)

        self.wave_init = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter * 2, 1, 1)).repeat(1, 1, self.image_size, self.image_size))
        self.wave_to_cell = DynaResidualU(self.lat_size * (2 if self.wavify_lat else 1), self.n_filter * 2, self.n_filter, self.n_layers)
        self.cell_to_cell = DynaResidualBlock(self.lat_size * (2 if self.wavify_lat else 1), self.n_filter * 3, self.n_filter * 2, self.n_filter * 2)
        self.wave_to_wave = DynaResidualU(self.lat_size * (2 if self.wavify_lat else 1), self.n_filter * 2, self.n_filter, self.n_layers)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.image_size + (1 if self.causal else 0), self.image_size + (1 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

        self.out_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 3, 1, 1)
        self.out_to_lat = nn.Linear(sum(self.conv_state_size), self.lat_size if not z_out else z_dim)

    def forward(self, x, labels):
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        inj_lat, _, _ = self.s_enc(x, labels)

        out = self.conv_img(x)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=x.device))

        out_embs = [out]

        l_wave = self.wave_init.repeat(batch_size, 1, 1, 1)
        l_wave = self.wave_to_cell(l_wave, inj_lat)

        wave = l_wave
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))

            out_new = torch.cat([out_new, complex_wave(wave)], dim=1)
            out_new = self.cell_to_cell(out_new, inj_lat)
            out_new, s_wave = torch.split(out_new, self.n_filter, dim=1)

            if c < self.n_calls - 1:
                wave = torch.cat([l_wave, s_wave], dim=1)
                wave = self.wave_to_wave(wave, inj_lat)

            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=x.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=x.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=x.device).to(float_type))
            out = out + (0.1 * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
            out_embs.append(out)

        conv_state = self.out_conv(out)
        if self.multi_cut:
            conv_state_f, conv_state_h, conv_state_w, conv_state_fh, conv_state_fw, conv_state_hw, conv_state_g = torch.split(conv_state, self.split_sizes, dim=1)
            conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
                                    conv_state_h.mean(dim=(1, 3)),
                                    conv_state_w.mean(dim=(1, 2)),
                                    conv_state_fh.mean(dim=3).view(batch_size, -1),
                                    conv_state_fw.mean(dim=2).view(batch_size, -1),
                                    conv_state_hw.mean(dim=1).view(batch_size, -1),
                                    conv_state_g.mean(dim=(1, 2, 3)).view(batch_size, 1)], dim=1)
        else:
            conv_state = conv_state.mean(dim=(2, 3))
        lat = self.out_to_lat(conv_state)

        if self.wavify_lat:
            lat = complex_wave(lat)

        return lat, out_embs, wave


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, env_feedback=False, multi_cut=True, auto_reg=False, ce_out=False, wavify_lat=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size * (2 if wavify_lat else 1)
        self.n_calls = n_calls
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.ce_out = ce_out
        self.n_layers = int(np.log2(image_size)) - 2

        self.wave_init = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter * 2, 1, 1)).repeat(1, 1, self.image_size, self.image_size))
        self.wave_to_cell = DynaResidualU(self.lat_size, self.n_filter * 2, self.n_filter, self.n_layers)
        self.cell_to_cell = DynaResidualBlock(self.lat_size, self.n_filter * 3, self.n_filter * 2, self.n_filter * 2)
        self.wave_to_wave = DynaResidualU(self.lat_size, self.n_filter * 2, self.n_filter, self.n_layers)

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

        out_embs = [out]

        l_wave = self.wave_init.repeat(batch_size, 1, 1, 1)
        l_wave = self.wave_to_cell(l_wave, lat)

        wave = l_wave
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))

            out_new = torch.cat([out_new, complex_wave(wave)], dim=1)
            out_new = self.cell_to_cell(out_new, lat)
            out_new, s_wave = torch.split(out_new, self.n_filter, dim=1)

            if c < self.n_calls - 1:
                wave = torch.cat([l_wave, s_wave], dim=1)
                wave = self.wave_to_wave(wave, lat)

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

        return out, out_embs, wave
