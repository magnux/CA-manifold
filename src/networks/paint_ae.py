import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.posencoding import sin_cos_pos_encoding_nd
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.residualblock import ResidualBlock
from src.layers.dynalinear import DynaLinear
from src.layers.dynaconv import DynaConv
from src.layers.dynaresidualblock import DynaResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np
from itertools import chain

# from src.networks.conv_ae import Encoder


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, perception_noise, fire_rate,
                 skip_fire=False, causal=False, gated=False, env_feedback=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, ce_in=False, **kwargs):
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
        self.ce_in = ce_in

        self.split_sizes = [self.n_filter] * 7 if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.image_size, self.image_size, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.n_filter]

        self.conv_img = nn.Conv2d(self.in_chan if not self.ce_in else self.in_chan * 256, self.n_filter, 1, 1, 0)

        self.register_buffer('pos_enc', sin_cos_pos_encoding_nd(self.image_size, 2))
        self.state_to_freq = DynaResidualBlock(self.lat_size, self.n_filter, self.pos_enc.size(1), self.n_filter * 2,
                                               lat_factor=2)
        self.freq_to_state = DynaConv(self.lat_size, self.pos_enc.size(1), self.n_filter)
                
        self.res_scales = int(np.log2(self.image_size)) - 1
        self.ms_convs = nn.ModuleList([ResidualBlock(self.n_filter * 2, self.n_filter) for _ in range(self.res_scales)])
        self.ms_to_state = ResidualBlock(self.n_filter * self.res_scales, self.n_filter * (2 if self.gated else 1))

        if self.env_feedback:
            self.feed_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
            self.feed_fc = LinearResidualBlock(sum(self.conv_state_size), z_dim, z_dim * 2)
            self.feed_dyn = DynaLinear(self.lat_size, z_dim, self.lat_size)

        if self.skip_fire:
            self.skip_fire_mask = torch.tensor(np.indices((1, 1, self.image_size + (2 if self.causal else 0), self.image_size + (2 if self.causal else 0))).sum(axis=0) % 2, requires_grad=False)

        self.out_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
        self.out_to_lat = LinearResidualBlock(sum(self.conv_state_size) + self.lat_size, self.lat_size, self.lat_size * 2)
        self.lat_to_lat = nn.Linear(self.lat_size, self.lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        if self.ce_in:
            x = x.view(batch_size, self.in_chan * 256, self.image_size, self.image_size)

        out = self.conv_img(x)

        if self.perception_noise and self.training:
            noise_mask = torch.round_(torch.rand([batch_size, 1], device=x.device))
            noise_mask = noise_mask * torch.round_(torch.rand([batch_size, self.n_calls], device=x.device))

        out_embs = [out]
        if self.env_feedback:
            conv_state = self.feed_conv(out)
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
            dyna_lat = self.feed_dyn(self.feed_fc(conv_state), inj_lat)
        else:
            dyna_lat = inj_lat
        lat = torch.zeros(batch_size, self.lat_size, device=x.device)
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))
            pos_encoding = self.state_to_freq(out, dyna_lat).mean(dim=(2, 3), keepdims=True)
            pos_encoding = pos_encoding * self.pos_enc.repeat(batch_size, 1, 1, 1)
            pos_encoding = self.freq_to_state(pos_encoding, dyna_lat)
            if not self.auto_reg:
                out_new = F.instance_norm(out_new)
                pos_encoding = F.instance_norm(pos_encoding)
            out_new = torch.cat([out_new, pos_encoding], dim=1)
            out_news = []
            for s in range(self.res_scales):
                conv_in = out_new
                if s > 0:
                    conv_in = F.interpolate(conv_in, scale_factor=1. / (2 ** s))
                conv_out = self.ms_convs[s](conv_in)
                if s > 0:
                    conv_out = F.interpolate(conv_out, size=out_new.shape[2:])
                out_news.append(conv_out)
            out_new = torch.cat(out_news, dim=1)
            out_new = self.ms_to_state(out_new)
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
            out = out_new
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
            lat = self.out_to_lat(torch.cat([conv_state, lat], dim=1))

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
                 skip_fire=False, log_mix_out=False, causal=False, gated=False, env_feedback=False, multi_cut=True, z_dim=0, auto_reg=False, ce_out=False, n_seed=1, **kwargs):
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

        self.register_buffer('seed', torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter, 1, 1).repeat(1, 1, self.image_size, self.image_size)))
        self.register_buffer('in_proj', torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter, 1, 1).repeat(1, 1, self.image_size, self.image_size)))

        self.register_buffer('pos_enc', sin_cos_pos_encoding_nd(self.image_size, 2))
        self.state_to_freq = DynaResidualBlock(self.lat_size, self.n_filter, self.pos_enc.size(1), self.n_filter * 2,
                                               lat_factor=2)
        self.freq_to_state = DynaConv(self.lat_size, self.pos_enc.size(1), self.n_filter)

        self.res_scales = int(np.log2(self.image_size)) - 1
        self.ms_convs = nn.ModuleList([ResidualBlock(self.n_filter * 2, self.n_filter) for _ in range(self.res_scales)])
        self.ms_to_state = ResidualBlock(self.n_filter * self.res_scales, self.n_filter * (2 if self.gated else 1))

        if self.env_feedback:
            self.feed_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
            self.feed_fc = LinearResidualBlock(sum(self.conv_state_size), z_dim, z_dim * 2)
            self.feed_dyn = DynaLinear(self.lat_size, z_dim, self.lat_size)

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
        if self.env_feedback:
            conv_state = self.feed_conv(out)
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
            dyna_lat = self.feed_dyn(self.feed_fc(conv_state), lat)
        else:
            dyna_lat = lat
        for c in range(self.n_calls):
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])
            out_new = out
            if self.perception_noise and self.training:
                out_new = out_new + (noise_mask[:, c].view(batch_size, 1, 1, 1) * 1e-2 * torch.randn_like(out_new))
            pos_encoding = self.state_to_freq(out, dyna_lat).mean(dim=(2, 3), keepdims=True)
            pos_encoding = pos_encoding * self.pos_enc.repeat(batch_size, 1, 1, 1)
            pos_encoding = self.freq_to_state(pos_encoding, dyna_lat)
            if not self.auto_reg:
                out_new = F.instance_norm(out_new)
                pos_encoding = F.instance_norm(pos_encoding)
            out_new = torch.cat([out_new, pos_encoding], dim=1)
            out_news = []
            for s in range(self.res_scales):
                conv_in = out_new
                if s > 0:
                    conv_in = F.interpolate(conv_in, scale_factor=1. / (2 ** s))
                conv_out = self.ms_convs[s](conv_in)
                if s > 0:
                    conv_out = F.interpolate(conv_out, size=out_new.shape[2:])
                out_news.append(conv_out)
            out_new = torch.cat(out_news, dim=1)
            out_new = self.ms_to_state(out_new)
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
