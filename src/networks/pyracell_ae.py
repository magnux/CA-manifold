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
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.irm import IRMConv
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed, checkerboard_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain

from src.networks.conv_ae import Encoder


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, shared_params, perception_noise, fire_rate,
                 causal=False, gated=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = int(np.ceil(np.log2(image_size) - np.log2(16)))
        self.shared_params = shared_params
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * 16, self.n_filter * 16, 16 ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_chan, self.n_filter, 1, 1, 0),
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
        )
        if self.causal:
            self.frac_irm = IRMConv(self.n_filter)
        self.frac_sobel = SinSobel(self.n_filter, 3, 1, left_sided=self.causal)
        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor)
                                            for _ in range(1 if self.shared_params else self.n_calls)])
        self.frac_dyna_conv = nn.ModuleList([
            DynaResidualBlock(lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                              self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter)
            for _ in range(1 if self.shared_params else self.n_calls)])

        self.frac_ds = nn.Sequential(
            GaussianSmoothing(n_filter, 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, scale_factor=0.5, mode='bilinear', align_corners=False)),
        )

        self.out_conv = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
            nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0),
        )
        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, lat_size if not z_out else z_dim)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        out = self.in_conv(x)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=x.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            if self.causal:
                out_new = self.frac_irm(out_new)
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            out_new = self.frac_dyna_conv[0 if self.shared_params else c](out_new, torch.cat([inj_lat, out_new.mean((2, 3))], 1) if self.env_feedback else inj_lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=x.device) <= self.fire_rate).to(float_type)
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.training and self.auto_reg:
                with torch.no_grad():
                    auto_reg_grad = (2 / out.numel()) * out.sign() * F.relu(out.abs() - 0.99)
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out = self.frac_ds(out)
            out_embs.append(out)

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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, shared_params, perception_noise, fire_rate,
                 log_mix_out=False, causal=False, gated=False, env_feedback=False, auto_reg=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = int(np.ceil(np.log2(image_size) - np.log2(16))) + 1
        self.shared_params = shared_params
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.auto_reg = auto_reg

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        self.in_conv = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
            GaussianSmoothing(self.n_filter, 3, 1, 1),
            LambdaLayer(lambda x: F.interpolate(x, size=16, mode='bilinear', align_corners=False)),
        )

        self.seed = nn.Parameter(ca_seed(1, self.n_filter, 16, 'cpu'))
        if self.causal:
            self.frac_irm = IRMConv(self.n_filter)
        self.frac_sobel = SinSobel(self.n_filter, 3, 1, left_sided=self.causal)
        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor)
                                            for _ in range(1 if self.shared_params else self.n_calls)])
        self.frac_dyna_conv = nn.ModuleList([
            DynaResidualBlock(self.lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                              self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter)
            for _ in range(1 if self.shared_params else self.n_calls)])

        self.frac_us = nn.Sequential(
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
            GaussianSmoothing(self.n_filter, 3, 1, 1)
        )

        self.out_conv = nn.Sequential(
            *([LambdaLayer(lambda x: F.interpolate(x, size=image_size, mode='bilinear', align_corners=False))] if np.mod(np.log2(image_size), 1) == 0 else []),
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 1, 1, 0),
        )

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device).to(float_type)
            out = torch.cat([self.seed.to(float_type)] * batch_size, 0)
        else:
            out = self.in_conv(ca_init)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            if self.causal:
                out_new = self.frac_irm(out_new)
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            out_new = self.frac_dyna_conv[0 if self.shared_params else c](out_new, torch.cat([lat, out_new.mean((2, 3))], 1) if self.env_feedback else lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(float_type)
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:]
            if self.training and self.auto_reg:
                with torch.no_grad():
                    auto_reg_grad = (2 / out.numel()) * out.sign() * F.relu(out.abs() - 0.99)
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            if c < self.n_calls - 1:
                out = self.frac_us(out)
            out_embs.append(out)

        out = self.out_conv(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw
