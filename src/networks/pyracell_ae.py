import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.gaussiansmoothing import GaussianSmoothing
from src.layers.noiseinjection import NoiseInjection
from src.layers.lambd import LambdaLayer
from src.layers.sobel import SinSobel
from src.layers.gaussgrads import GaussGrads
from src.layers.dynaresidualblock import DynaResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed, checkerboard_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, shared_params, perception_noise, fire_rate,
                 causal=False, gated=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, ce_in=False, gauss_grads=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_layers = (int(np.ceil(np.log2(image_size) - np.log2(16))) + 1)
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.ce_in = ce_in

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * 16, self.n_filter * 16, 16 ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, self.n_filter, 1, 1, 0)

        if gauss_grads:
            self.frac_sobel = GaussGrads(self.n_filter, 3, 1)
        else:
            self.frac_sobel = SinSobel(self.n_filter, 3, 1, left_sided=self.causal)

        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor)
                                            for _ in range(1 if self.shared_params else self.n_layers)])
        self.frac_dyna_conv = nn.ModuleList([
            DynaResidualBlock(lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                              self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter)
            for _ in range(1 if self.shared_params else self.n_layers)])
        self.frac_conv = nn.ModuleList([
            ResidualBlock(self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter, 1, 1, 0)
            for _ in range(1 if self.shared_params else self.n_layers)])

        self.frac_lat_exp = nn.ModuleList([nn.Linear(self.lat_size, self.lat_size) for _ in range(self.n_layers * n_calls)])

        self.frac_ds = nn.Sequential(
            GaussianSmoothing(n_filter, 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False, recompute_scale_factor=True))
        )

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16))
        self.out_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, lat_size if not z_out else z_dim)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        out = self.in_conv(x)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_layers * self.n_calls], device=x.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_layers * self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c // self.n_calls](out_new)
            out_new_f = self.frac_conv[0 if self.shared_params else c // self.n_calls](out_new)
            lat_new = torch.cat([self.frac_lat_exp[c](inj_lat), out_new.mean((2, 3))], 1) if self.env_feedback else self.frac_lat_exp[c](inj_lat)
            out_new_d = self.frac_dyna_conv[0 if self.shared_params else c // self.n_calls](out_new, lat_new)
            out_new = out_new_f + out_new_d
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=x.device) <= self.fire_rate).to(float_type)
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            if c < (self.n_layers * self.n_calls) - 1 and c % self.n_calls == self.n_calls - 1:
                out = self.frac_ds(out)
            out_embs.append(out)

        out = out + torch.cat([self.seed.to(float_type)] * batch_size, 0)
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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, shared_params, perception_noise, fire_rate,
                 log_mix_out=False, causal=False, gated=False, env_feedback=False, auto_reg=False, ce_out=False, n_seed=1, gauss_grads=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_layers = (int(np.ceil(np.log2(image_size) - np.log2(16))) + 1)
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.auto_reg = auto_reg
        self.ce_out = ce_out
        self.n_seed = n_seed

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter)).reshape(n_seed, self.n_filter, 1, 1))
        self.in_ds = nn.Sequential(
            GaussianSmoothing(self.n_filter, 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, size=16, mode='bilinear', align_corners=False)),
        )

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, 16, 16))

        if gauss_grads:
            self.frac_sobel = GaussGrads(self.n_filter, 3, 1)
        else:
            self.frac_sobel = SinSobel(self.n_filter, 3, 1, left_sided=self.causal)

        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor)
                                            for _ in range(1 if self.shared_params else self.n_layers)])
        self.frac_dyna_conv = nn.ModuleList([
            DynaResidualBlock(self.lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                              self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter)
            for _ in range(1 if self.shared_params else self.n_layers)])
        self.frac_conv = nn.ModuleList([
            ResidualBlock(self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter, 1, 1, 0)
            for _ in range(1 if self.shared_params else self.n_layers)])

        self.frac_lat_exp = nn.ModuleList([nn.Linear(self.lat_size, self.lat_size) for _ in range(self.n_layers * n_calls)])
        self.frac_noise = nn.ModuleList([NoiseInjection(n_filter) for _ in range(self.n_layers * n_calls)])

        self.frac_us = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GaussianSmoothing(self.n_filter, 3, 1, 1)
        )

        if self.log_mix_out:
            out_f = 10 * ((self.out_chan * 3) + 1)
        elif self.ce_out:
            out_f = self.out_chan * 256
            ce_pos = torch.arange(0, 256).view(1, 256, 1, 1, 1)
            ce_pos = ce_pos.expand(-1, -1, self.out_chan, self.image_size, self.image_size)
            self.register_buffer('ce_pos', ce_pos)
        else:
            out_f = self.out_chan
        self.out_conv = nn.Sequential(
            *([LambdaLayer(lambda x: F.interpolate(x, size=image_size, mode='bilinear', align_corners=False))] if np.mod(np.log2(image_size), 1) == 0 else []),
            nn.Conv2d(self.n_filter, out_f, 1, 1, 0),
        )

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
            out = torch.cat([seed.to(float_type)] * batch_size, 0)
        else:
            if isinstance(seed_n, tuple):
                proj = self.in_proj[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                proj = self.in_proj[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                proj = self.in_proj[seed_n:seed_n + 1, ...]
            proj = torch.cat([proj.to(float_type)] * batch_size, 0)
            out = self.in_ds(ca_init) + proj

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_layers * self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_layers * self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c // self.n_calls](out_new)
            out_new_f = self.frac_conv[0 if self.shared_params else c // self.n_calls](out_new)
            lat_new = torch.cat([self.frac_lat_exp[c](lat), out_new.mean((2, 3))], 1) if self.env_feedback else self.frac_lat_exp[c](lat)
            out_new_d = self.frac_dyna_conv[0 if self.shared_params else c // self.n_calls](out_new, lat_new)
            out_new = out_new_f + out_new_d
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(float_type)
            out_new = self.frac_noise[c](out_new)
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            if c < (self.n_layers * self.n_calls) - 1 and c % self.n_calls == self.n_calls - 1:
                out = self.frac_us(out)
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

        return out, out_embs, out_raw
