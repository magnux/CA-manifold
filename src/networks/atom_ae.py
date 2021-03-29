import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.gaussiansmoothing import GaussianSmoothing
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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.n_calls = n_calls
        self.lat_size = lat_size if lat_size > 3 else 512
        self.auto_reg = auto_reg

        self.in_conv = ResidualBlock(self.in_chan, self.n_filter, None, 1, 1, 0)
        frac_sobel = SinSobel(self.in_chan, [(2 ** i) + 1 for i in range(1, int(np.log2(self.image_size) - 1), 1)],
                              [2 ** (i - 1) for i in range(1, int(np.log2(self.image_size) - 1), 1)])
        self.frac_sobel = nn.Sequential(
            nn.Conv2d(self.n_filter, self.in_chan, 1, 1, 0),
            frac_sobel,
            nn.Conv2d(self.in_chan * frac_sobel.c_factor, self.n_filter, 1, 1, 0),
        )
        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm2d(self.n_filter)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter, self.n_filter, self.n_filter * 2, norm_weights=True)

        self.out_conv = ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0)

        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(self.n_filter, self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, lat_size if not z_out else z_dim)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)
        float_type = torch.float16 if isinstance(x, torch.cuda.HalfTensor) else torch.float32

        out = self.in_conv(x)

        out_embs = [out]
        auto_reg_grads = []
        for _ in range(self.n_calls):
            out_new = out * self.frac_sobel(out)
            if not self.auto_reg:
                out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, inj_lat)
            out = out + (0.1 * out_new)
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out_embs.append(out)

        out = self.out_conv(out)
        lat = self.out_to_lat(out.mean(dim=(2, 3)))

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
    def __init__(self, n_labels, lat_size, image_size, channels, n_filter, n_calls, log_mix_out=False, auto_reg=False, n_seed=1, **kwargs):
        super().__init__()
        self.out_chan = channels
        self.n_labels = n_labels
        self.image_size = image_size
        self.n_filter = n_filter
        self.n_calls = n_calls
        self.lat_size = lat_size
        self.log_mix_out = log_mix_out
        self.auto_reg = auto_reg

        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter * self.n_filter)).reshape(n_seed, self.n_filter, self.n_filter))

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.image_size, self.image_size))
        frac_sobel = SinSobel(self.out_chan, [(2 ** i) + 1 for i in range(1, int(np.log2(self.image_size)-1), 1)],
                              [2 ** (i - 1) for i in range(1, int(np.log2(self.image_size)-1), 1)])
        self.frac_sobel = nn.Sequential(
                nn.Conv2d(self.n_filter, self.out_chan, 1, 1, 0),
                frac_sobel,
                nn.Conv2d(self.out_chan * frac_sobel.c_factor, self.n_filter, 1, 1, 0),
        )
        if not self.auto_reg:
            self.frac_norm = nn.InstanceNorm2d(self.n_filter)
        self.frac_dyna_conv = DynaResidualBlock(self.lat_size, self.n_filter, self.n_filter, self.n_filter * 2, norm_weights=True)

        self.out_conv = nn.Sequential(
            *([LambdaLayer(lambda x: F.interpolate(x, size=image_size, mode='bilinear', align_corners=False))] if np.mod(np.log2(image_size), 1) == 0 else []),
            ResidualBlock(self.n_filter, self.n_filter, None, 1, 1, 0),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 1, 1, 0),
        )

    def forward(self, lat, ca_init=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
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
            out = ca_init.permute(0, 2, 3, 1).reshape(batch_size, self.image_size * self.image_size, self.n_filter)
            out = torch.bmm(out, proj).reshape(batch_size, self.image_size, self.image_size, self.n_filter).permute(0, 3, 1, 2).contiguous()

        out_embs = [out]
        auto_reg_grads = []
        for _ in range(self.n_calls):
            out_new = out * self.frac_sobel(out)
            if not self.auto_reg:
                out_new = self.frac_norm(out_new)
            out_new = self.frac_dyna_conv(out_new, lat)
            out = out + (0.1 * out_new)
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out_embs.append(out)

        out = self.out_conv(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw
