import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.imagescaling import DownScale, UpScale
from src.layers.lambd import LambdaLayer
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.centroids import Centroids
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, gated=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = n_calls * 4
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.ds_size, self.n_filter * self.ds_size, self.ds_size ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_chan, self.n_filter, 3, 1, 1),
            *([DownScale(self.n_filter, self.n_filter, self.image_size, self.ds_size)] if self.ds_size < self.image_size else []),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.ds_size))] if self.ds_size < self.image_size else []),
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
        )
        self.in_norm = nn.Sequential(
            nn.InstanceNorm2d(self.n_filter),
            Centroids(self.n_filter, 2 ** 10),
        )

        self.frac_sobel = SinSobel(self.n_filter, 5, 2, left_sided=causal)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)
        self.frac_dyna_conv = DynaResidualBlock(lat_size + (n_filter * 3 if self.env_feedback else 0), self.n_filter * 3, self.n_filter * (2 if self.gated else 1), self.n_filter)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.ds_size + (2 if self.causal else 0), self.ds_size + (2 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

        self.out_norm = nn.InstanceNorm2d(self.n_filter)
        self.out_conv = ResidualBlock(self.n_filter, sum(self.split_sizes), None, 1, 1, 0)
        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            # *([] if lat_size > 3 else [nn.Linear(self.lat_size, lat_size)]),
            nn.Linear(self.lat_size, lat_size if not z_out else z_dim)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        out = self.in_conv(x)
        out = self.in_norm(out)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=x.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 2, 0, 2])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, torch.cat([inj_lat, out_new.mean((2, 3))], 1) if self.env_feedback else inj_lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=x.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=x.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=x.device).to(float_type))
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 2:, 2:]
            out_embs.append(out)

        out = self.out_norm(out)
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


class ZInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        kwargs['z_out'] = True
        super().__init__(**kwargs)


class Decoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, gated=False, env_feedback=False, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls * 4
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.log_mix_out = log_mix_out
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        # self.seed = nn.Parameter(ca_seed(1, self.n_filter, self.ds_size, 'cpu', all_channels=True))

        self.frac_sobel = SinSobel(self.n_filter, 5, 2, left_sided=causal)
        self.frac_norm = nn.InstanceNorm2d(self.n_filter * 3)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size + (n_filter * 3 if self.env_feedback else 0), self.n_filter * 3, self.n_filter * (2 if self.gated else 1), self.n_filter)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.ds_size + (2 if self.causal else 0), self.ds_size + (2 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

        self.out_norm = nn.Sequential(
            nn.InstanceNorm2d(self.n_filter),
            Centroids(self.n_filter, 2 ** 10),
        )
        self.out_conv = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            out = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
            # out = torch.cat([self.seed.to(float_type)] * batch_size, 0)
        else:
            out = ca_init

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 2, 0, 2])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * torch.randn_like(out_new))
            out_new = self.frac_sobel(out_new)
            out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, torch.cat([lat, out_new.mean((2, 3))], 1) if self.env_feedback else lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=lat.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=lat.device).to(float_type))
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 2:, 2:]
            out_embs.append(out)

        out = self.out_norm(out)
        out = self.out_conv(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw
