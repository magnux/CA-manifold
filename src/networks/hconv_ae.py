import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.networks.base import LabsEncoder
from src.layers.randgrads import RandGrads
from src.layers.posencoding import cos_pos_encoding_dyn
from src.utils.model_utils import ca_seed, checkerboard_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic
import numpy as np


class Encoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, shared_params,
                 injected=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = injected
        self.multi_cut = multi_cut
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.auto_reg = auto_reg
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.image_size, self.n_filter * self.image_size, self.image_size ** 2] if self.multi_cut else [self.n_filter]

        self.in_conv = nn.Conv2d(self.in_chan, self.n_filter, 3, 1, 1)

        self.frac_sobel = RandGrads(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(image_size)-1), 1)],
                                                   [2 ** (i - 1) for i in range(1, int(np.log2(image_size)-1), 1)], n_calls=n_calls)
        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor) for _ in range(1 if self.shared_params else self.n_calls)])

        self.register_buffer('frac_pos', cos_pos_encoding_dyn(self.image_size, 2, self.n_calls))
        self.frac_conv = nn.ModuleList([ResidualBlock(self.n_filter * self.frac_sobel.c_factor + (self.frac_pos.shape[2] if self.injected else 0), self.n_filter, self.n_filter * 4, 1, 1, 0) for _ in range(1 if self.shared_params else self.n_calls)])

        if self.injected:
            self.inj_cond = LinearResidualBlock(self.lat_size, self.frac_pos.shape[2] * (1 if self.shared_params else self.n_calls))

        self.out_conv = nn.Conv2d(self.n_filter, sum(self.split_sizes), 1, 1, 0)
        self.out_to_lat = nn.Linear(sum(self.conv_state_size), lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = self.in_conv(x)

        if self.injected:
            cond_factors = self.inj_cond(inj_lat)
            cond_factors = torch.split(cond_factors, self.frac_pos.shape[2], dim=1)

        out_embs = [out]
        self.frac_sobel.call_c = 0
        for c in range(self.n_calls):
            out_new = self.frac_sobel(out)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            if self.injected:
                c_fact = cond_factors[0 if self.shared_params else c].view(batch_size, self.frac_pos.shape[2], 1, 1).contiguous().repeat(1, 1, self.image_size, self.image_size)
                c_fact = self.frac_pos[c, ...] * c_fact
                out_new = torch.cat([out_new, c_fact], dim=1)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out_new
            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
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


class InjectedEncoder(Encoder):
    def __init__(self, **kwargs):
        kwargs['injected'] = True
        super().__init__(**kwargs)


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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, shared_params,
                 log_mix_out=False, auto_reg=False, ce_out=False, n_seed=1, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.out_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.log_mix_out = log_mix_out
        self.auto_reg = auto_reg
        self.ce_out = ce_out
        self.n_seed = n_seed

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.image_size, self.image_size))
        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(self.n_seed, self.n_filter)).reshape(self.n_seed, self.n_filter, 1, 1))

        self.frac_sobel = RandGrads(self.n_filter, [(2 ** i) + 1 for i in range(1, int(np.log2(image_size)-1), 1)],
                                                   [2 ** (i - 1) for i in range(1, int(np.log2(image_size)-1), 1)], n_calls=n_calls)
        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter * self.frac_sobel.c_factor) for _ in range(1 if self.shared_params else self.n_calls)])
        self.register_buffer('frac_pos', cos_pos_encoding_dyn(self.image_size, 2, self.n_calls))
        self.frac_conv = nn.ModuleList([ResidualBlock(self.n_filter * self.frac_sobel.c_factor + self.frac_pos.shape[2], self.n_filter, self.n_filter * 4, 1, 1, 0) for _ in range(1 if self.shared_params else self.n_calls)])

        self.inj_cond = LinearResidualBlock(self.lat_size, self.frac_pos.shape[2] * (1 if self.shared_params else self.n_calls))

        self.out_conv = nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1)

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        cond_factors = self.inj_cond(lat)
        cond_factors = torch.split(cond_factors, self.frac_pos.shape[2], dim=1)

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
            proj = torch.cat([proj.to(float_type)] * batch_size, 0)
            out = ca_init + proj

        out_embs = [out]
        self.frac_sobel.call_c = 0
        for c in range(self.n_calls):
            out_new = self.frac_sobel(out)
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            c_fact = cond_factors[0 if self.shared_params else c].view(batch_size, self.frac_pos.shape[2], 1, 1).contiguous().repeat(1, 1, self.image_size, self.image_size)
            c_fact = self.frac_pos[c, ...] * c_fact
            out_new = torch.cat([out_new, c_fact], dim=1)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out_new
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

