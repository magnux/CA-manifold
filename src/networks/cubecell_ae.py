import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.lambd import LambdaLayer
from src.layers.sobel import SinSobel
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.irm import IRMConv
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, causal=False, gated=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, conv_irm=False, ce_in=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = n_calls
        self.perception_noise = perception_noise
        self.fire_rate = fire_rate
        self.skip_fire = skip_fire
        assert not self.fire_rate < 1.0 or not skip_fire, "fire_rate and skip_fire are mutually exclusive options"
        self.causal = causal
        self.gated = gated
        self.env_feedback = env_feedback
        self.multi_cut = multi_cut
        self.auto_reg = auto_reg
        self.conv_irm = conv_irm
        self.ce_in = ce_in

        self.cube_size = int(np.ceil((self.image_size ** 2) ** (1/3)))
        self.pad_size = int((self.cube_size ** 3) - (self.image_size ** 2))
        self.register_buffer('cube_pad', torch.zeros(1, self.n_filter, self.pad_size))

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Sequential(
            nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, self.n_filter, 1, 1, 0),
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
        )
        if self.conv_irm:
            self.frac_irm = IRMConv(self.n_filter, dim=3)
        self.frac_sobel = SinSobel(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(self.cube_size)-1), 1)],
                                                  [2 ** (i - 1) for i in range(1, int(np.log2(self.cube_size)-1), 1)], dim=3, left_sided=self.causal)
        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm3d(self.n_filter * self.frac_sobel.c_factor)
        self.frac_dyna_conv = DynaResidualBlock(lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                                                self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter, dim=3)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.image_size + (1 if self.causal else 0),
                                                                 self.image_size + (1 if self.causal else 0),
                                                                 self.image_size + (1 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

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

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        out = self.in_conv(x)

        out = torch.cat([out.view(batch_size, self.n_filter, -1), self.cube_pad.expand(batch_size, -1, -1)], dim=2)
        out = out.view(batch_size, self.n_filter, self.cube_size, self.cube_size, self.cube_size)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=x.device))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1, 1) * torch.randn_like(out_new))
            if self.conv_irm:
                out_new = self.frac_irm(out_new)
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, torch.cat([inj_lat, out_new.mean((2, 3, 4))], 1) if self.env_feedback else inj_lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3), out.size(4)], device=x.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=x.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=x.device).to(float_type))
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2 / out.numel()) * out.sign() * F.relu(out.abs() - 0.99)
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out_embs.append(out)

        out = out.view(batch_size, self.n_filter, -1)
        out = out[:, :, :self.image_size ** 2]
        out = out.view(batch_size, self.n_filter, self.image_size, self.image_size)

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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, log_mix_out=False, causal=False, gated=False, env_feedback=False, redec_ap=False, auto_reg=False, conv_irm=False, ce_out=False, n_seed=1, **kwargs):
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
        self.redec_ap = redec_ap
        self.auto_reg = auto_reg
        self.conv_irm = conv_irm
        self.ce_out = ce_out

        self.cube_size = int(np.ceil((self.image_size ** 2) ** (1/3)))
        self.pad_size = int((self.cube_size ** 3) - (self.image_size ** 2))
        self.register_buffer('cube_pad', torch.zeros(1, self.n_filter, self.pad_size))

        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)

        if self.redec_ap:
            self.in_ap = nn.AvgPool2d(5, 1, 2, count_include_pad=False)
        self.in_conv = ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0)

        self.seed = nn.ParameterList([nn.Parameter(ca_seed(1, self.n_filter, self.image_size, 'cpu')) for _ in range(n_seed)])
        if self.conv_irm:
            self.frac_irm = IRMConv(self.n_filter, dim=3)
        self.frac_sobel = SinSobel(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(self.cube_size)-1), 1)],
                                                  [2 ** (i - 1) for i in range(1, int(np.log2(self.cube_size)-1), 1)], left_sided=self.causal, dim=3)
        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm3d(self.n_filter * self.frac_sobel.c_factor)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size + (self.n_filter * self.frac_sobel.c_factor if self.env_feedback else 0),
                                                self.n_filter * self.frac_sobel.c_factor, self.n_filter * (2 if self.gated else 1), self.n_filter, dim=3)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.image_size + (2 if self.causal else 0),
                                                                 self.image_size + (2 if self.causal else 0),
                                                                 self.image_size + (2 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

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
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
            nn.Conv2d(self.n_filter, out_f, 1, 1, 0),
        )

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device).to(float_type)
            out = torch.cat([self.seed[seed_n].to(float_type)] * batch_size, 0)
        else:
            if self.redec_ap:
                ca_init = self.in_ap(ca_init)
            out = self.in_conv(ca_init)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out = torch.cat([out.view(batch_size, self.n_filter, -1), self.cube_pad.expand(batch_size, -1, -1)], dim=2)
        out = out.view(batch_size, self.n_filter, self.cube_size, self.cube_size, self.cube_size)

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1, 1) * torch.randn_like(out_new))
            if self.conv_irm:
                out_new = self.frac_irm(out_new)
            out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, torch.cat([lat, out_new.mean((2, 3, 4))], 1) if self.env_feedback else lat)
            if self.gated:
                out_new, out_new_gate = torch.split(out_new, self.n_filter, dim=1)
                out_new = out_new * torch.sigmoid(out_new_gate)
            if self.fire_rate < 1.0:
                out_new = out_new * (torch.rand([batch_size, 1, out.size(2), out.size(3), out.size(4)], device=lat.device) <= self.fire_rate).to(float_type)
            if self.skip_fire:
                if c % 2 == 0:
                    out_new = out_new * self.skip_fire_mask.to(device=lat.device).to(float_type)
                else:
                    out_new = out_new * (1 - self.skip_fire_mask.to(device=lat.device).to(float_type))
            out = out + (leak_factor * out_new)
            if self.causal:
                out = out[:, :, 1:, 1:, 1:]
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2 / out.numel()) * out.sign() * F.relu(out.abs() - 0.99)
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out_embs.append(out)

        out = out.view(batch_size, self.n_filter, -1)
        out = out[:, :, :self.image_size ** 2]
        out = out.view(batch_size, self.n_filter, self.image_size, self.image_size)

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
