import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.posencoding import sin_cos_pos_encoding_dyn
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.mixconv import MixConv
from src.layers.mixresidualblock import MixResidualBlock
from src.layers.randgrads import RandGrads
from src.layers.dynaconv import DynaConv
from src.layers.dynaresidualblock import DynaResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, shared_params,
                 multi_cut=True, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = True
        self.multi_cut = multi_cut
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.auto_reg = auto_reg

        self.split_sizes = [self.n_filter] * 7 if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.image_size, self.image_size, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.n_filter]

        self.conv_img = nn.Conv2d(self.in_chan, self.n_filter, 1, 1, 0)

        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm2d(self.n_filter)

        self.frac_conv = nn.ModuleList([MixResidualBlock(self.lat_size, self.n_filter, self.n_filter, self.n_filter * 4, lat_factor=4) for _ in range(1 if self.shared_params else self.n_calls)])

        self.out_conv = nn.ModuleList([MixConv(self.lat_size, self.n_filter, sum(self.split_sizes)) for _ in range(1 if self.shared_params else self.n_calls)])
        self.out_to_lat = nn.ModuleList([LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2) for _ in range(1 if self.shared_params else self.n_calls)])
        self.lat_to_lat = nn.Linear(self.lat_size, self.lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = self.conv_img(x)

        out_embs = [out]
        lat = inj_lat if inj_lat is not None else 0
        for c in range(self.n_calls):
            if not self.auto_reg:
                out = self.frac_norm(out)
            out = self.frac_conv[0 if self.shared_params else c](out, inj_lat)
            if self.multi_cut:
                conv_state = self.out_conv[0 if self.shared_params else c](out, inj_lat)
                conv_state_f, conv_state_h, conv_state_w, conv_state_fh, conv_state_fw, conv_state_hw, conv_state_g = torch.split(conv_state, self.split_sizes, dim=1)
                conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
                                        conv_state_h.mean(dim=(1, 3)),
                                        conv_state_w.mean(dim=(1, 2)),
                                        conv_state_fh.mean(dim=3).view(batch_size, -1),
                                        conv_state_fw.mean(dim=2).view(batch_size, -1),
                                        conv_state_hw.mean(dim=1).view(batch_size, -1),
                                        conv_state_g.mean(dim=(1, 2, 3)).view(batch_size, 1)], dim=1)
            else:
                conv_state = out.mean(dim=(2, 3))

            lat = lat + self.out_to_lat[0 if self.shared_params else c](conv_state)

            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
            out_embs.append(out)

        lat = self.lat_to_lat(lat)

        return lat, out_embs, None


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        self.inj_lat = self.labs_encoder(labels)
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
                 skip_fire=False, log_mix_out=False, causal=False, gated=False, env_feedback=False, multi_cut=True, auto_reg=False, ce_out=False, n_seed=1, **kwargs):
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
        self.n_seed = n_seed

        # self.split_sizes = [self.n_filter] * 7 if self.multi_cut else [self.n_filter]
        # self.conv_state_size = [self.n_filter, self.image_size, self.image_size, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.n_filter]
        # self.lat_to_in = LinearResidualBlockX(self.lat_size, sum(self.conv_state_size))
        # self.in_conv = nn.Conv2d(sum(self.split_sizes), self.n_filter, 1, 1, 0)
        # self.rein_conv = nn.Conv2d(self.n_filter, self.n_filter, 1, 1, 0)

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter, 1, 1).repeat(1, 1, self.image_size, self.image_size)))
        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter, 1, 1).repeat(1, 1, self.image_size, self.image_size)))

        # self.frac_sobel = RandGrads(self.n_filter, np.repeat([(2 ** i) + 1 for i in range(1, int(np.log2(image_size)-1), 1)], 3),
        #                                            np.repeat([2 ** (i - 1) for i in range(1, int(np.log2(image_size)-1), 1)], 3), n_calls=n_calls)
        # self.frac_factor = self.frac_sobel.c_factor
        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm2d(self.n_filter)
        self.register_buffer('frac_pos', sin_cos_pos_encoding_dyn(self.image_size, 2, self.n_calls))
        self.frac_wave = DynaConv(self.lat_size, self.frac_pos.size(2), self.n_filter)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter * 2, self.n_filter * (2 if self.gated else 1), self.n_filter * 2, lat_factor=2)

        # self.frac_lat = LinearResidualBlock(self.lat_size, self.lat_size)
        if self.env_feedback:
            self.frac_feedback = nn.Linear(sum(self.conv_state_size) + self.lat_size, self.lat_size)
            self.feed_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)

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

        self.out_conv = nn.Conv2d(self.n_filter, out_f, 1, 1, 0)

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.image_size, lat.device).to(float_type)
            if isinstance(seed_n, tuple):
                seed = self.seed[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                seed = self.seed[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                seed = self.seed[seed_n:seed_n + 1, ...]
            out = seed.to(float_type).repeat(batch_size, 1, 1, 1)
        else:
            if isinstance(seed_n, tuple):
                proj = self.in_proj[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                proj = self.in_proj[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                proj = self.in_proj[seed_n:seed_n + 1, ...]
            proj = proj.to(float_type).repeat(batch_size, 1, 1, 1)
            out = ca_init + proj

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=lat.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=lat.device))

        out_embs = [out]
        dyna_lat = lat
        # self.frac_sobel.call_c = 0
        for c in range(self.n_calls):
            if self.env_feedback:
                if self.multi_cut:
                    conv_state_f, conv_state_h, conv_state_w, conv_state_fh, conv_state_fw, conv_state_hw, conv_state_g = torch.split(self.feed_conv(out), self.split_sizes, dim=1)
                    conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
                                            conv_state_h.mean(dim=(1, 3)),
                                            conv_state_w.mean(dim=(1, 2)),
                                            conv_state_fh.mean(dim=3).view(batch_size, -1),
                                            conv_state_fw.mean(dim=2).view(batch_size, -1),
                                            conv_state_hw.mean(dim=1).view(batch_size, -1),
                                            conv_state_g.mean(dim=(1, 2, 3)).view(batch_size, 1)], dim=1)
                else:
                    conv_state = out.mean(dim=(2, 3))
                dyna_lat = self.frac_feedback(torch.cat([conv_state, dyna_lat], 1))
            # dyna_lat = self.frac_lat(dyna_lat)
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))
            # out_new = self.frac_sobel(out_new)
            if not self.auto_reg:
                out_new = self.frac_norm(out_new)
            pos_encoding = self.frac_pos[c].repeat(batch_size, 1, 1, 1)
            pos_encoding = self.frac_wave(pos_encoding, dyna_lat)
            out_new = torch.cat([out_new, pos_encoding], dim=1)
            out_new = self.frac_dyna_conv(out_new, dyna_lat)
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
            out = out_new
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

        return out, out_embs, out_raw
