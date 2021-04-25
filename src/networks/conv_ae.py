import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.imagescaling import DownScale, UpScale
from src.layers.lambd import LambdaLayer
from src.layers.dynaconv import DynaConv
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed, checkerboard_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic


class Encoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, shared_params,
                 injected=False, adain=False, dyncin=False, multi_cut=True, z_out=False, z_dim=0, auto_reg=False, **kwargs):
        super().__init__()
        self.injected = injected
        self.adain = adain
        self.dyncin = dyncin
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
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.split_sizes = [self.n_filter, self.n_filter, self.n_filter, 1] if self.multi_cut else [self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.ds_size, self.n_filter * self.ds_size, self.ds_size ** 2] if self.multi_cut else [self.n_filter]

        self.conv_img = nn.Sequential(
            nn.Conv2d(self.in_chan, self.n_filter, 3, 1, 1),
            *([DownScale(self.n_filter, self.n_filter, self.image_size, self.ds_size)] if self.ds_size < self.image_size else []),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.ds_size))] if self.ds_size < self.image_size else []),
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
        )

        if self.injected:
            if self.adain:
                self.inj_cond = nn.Sequential(
                    LinearResidualBlock(self.lat_size, self.lat_size),
                    LinearResidualBlock(self.lat_size, self.n_filter * 2 * (1 if self.shared_params else self.n_calls), self.lat_size * 2),
                )
            elif self.dyncin:
                self.inj_cond = nn.ModuleList([DynaConv(self.lat_size, self.n_filter, self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
            else:
                self.inj_cond = nn.Sequential(
                    LinearResidualBlock(self.lat_size, self.lat_size),
                    LinearResidualBlock(self.lat_size, self.n_filter * (1 if self.shared_params else self.n_calls), self.lat_size * 2),
                )
        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        conv_in_size = (1 if not self.injected or self.adain or self.dyncin else 2)
        self.frac_conv = nn.ModuleList([nn.Sequential(
                                            ResidualBlock(self.n_filter * conv_in_size, self.n_filter, self.n_filter * 4, 1, 1, 0),
                                            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1)
                                        ) for _ in range(1 if self.shared_params else self.n_calls)])

        self.out_conv = ResidualBlock(self.n_filter, sum(self.split_sizes), None, 1, 1, 0)
        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, self.lat_size if not z_out else z_dim)
        )

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = self.conv_img(x)

        if self.injected:
            if self.adain:
                cond_factors = self.inj_cond(inj_lat)
                cond_factors = torch.split(cond_factors, self.n_filter * 2, dim=1)
            elif self.dyncin:
                pass
            else:
                cond_factors = self.inj_cond(inj_lat)
                cond_factors = torch.split(cond_factors, self.n_filter, dim=1)

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            out_new = out
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            if self.injected:
                if self.adain:
                    s_fact, b_fact = torch.split(cond_factors[0 if self.shared_params else c], self.n_filter, dim=1)
                    s_fact = s_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                    b_fact = b_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                    out_new = (s_fact * out_new) + b_fact
                elif self.dyncin:
                    out_new = self.inj_cond[0 if self.shared_params else c](out_new, inj_lat)
                else:
                    c_fact = cond_factors[0 if self.shared_params else c].view(batch_size, self.n_filter, 1, 1).contiguous().repeat(1, 1, self.ds_size, self.ds_size)
                    out_new = torch.cat([out_new, c_fact], dim=1)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out + (leak_factor * out_new)
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
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
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, shared_params,
                 adain=False, dyncin=False, log_mix_out=False, redec_ap=False, auto_reg=False, n_seed=1, **kwargs):
        super().__init__()
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.out_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.n_calls = n_calls
        self.shared_params = shared_params
        self.adain = adain
        self.dyncin = dyncin
        self.log_mix_out = log_mix_out
        self.leak_factor = nn.Parameter(torch.ones([]) * 0.1)
        self.redec_ap = redec_ap
        self.auto_reg = auto_reg

        if not self.auto_reg:
            self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        conv_in_size = (1 if self.adain or self.dyncin else 2)
        self.frac_conv = nn.ModuleList([nn.Sequential(
                                            ResidualBlock(self.n_filter * conv_in_size, self.n_filter, self.n_filter * 4, 1, 1, 0),
                                            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1)
                                        ) for _ in range(1 if self.shared_params else self.n_calls)])

        self.seed = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter)).unsqueeze(2).unsqueeze(3).repeat(1, 1, self.ds_size, self.ds_size))

        if self.adain:
            self.inj_cond = nn.Sequential(
                LinearResidualBlock(self.lat_size, self.lat_size),
                LinearResidualBlock(self.lat_size, self.n_filter * 2 * (1 if self.shared_params else self.n_calls), self.lat_size * 2),
            )
        elif self.dyncin:
            self.inj_cond = nn.ModuleList([DynaConv(self.lat_size, self.n_filter, self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        else:
            self.inj_cond = nn.Sequential(
                LinearResidualBlock(self.lat_size, self.lat_size),
                LinearResidualBlock(self.lat_size, self.n_filter * (1 if self.shared_params else self.n_calls), self.lat_size * 2),
            )

        if self.redec_ap:
            self.in_ap = nn.AvgPool2d(5, 1, 2, count_include_pad=False)
        self.in_proj = nn.Parameter(torch.nn.init.orthogonal_(torch.empty(n_seed, self.n_filter * self.n_filter)).reshape(n_seed, self.n_filter, self.n_filter))

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None, ca_noise=None, seed_n=0):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if self.adain:
            cond_factors = self.inj_cond(lat)
            cond_factors = torch.split(cond_factors, self.n_filter * 2, dim=1)
        elif self.dyncin:
            pass
        else:
            cond_factors = self.inj_cond(lat)
            cond_factors = torch.split(cond_factors, self.n_filter, dim=1)

        if ca_init is None:
            # out = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
            if isinstance(seed_n, tuple):
                seed = self.seed[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                seed = self.seed[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                seed = self.seed[seed_n:seed_n + 1, ...]
            out = torch.cat([seed.to(float_type)] * batch_size, 0)
            if ca_noise is not None:
                out = out + ca_noise
        else:
            if self.redec_ap:
                ca_init = self.in_ap(ca_init)
            if isinstance(seed_n, tuple):
                proj = self.in_proj[seed_n[0]:seed_n[1], ...].mean(dim=0, keepdim=True)
            elif isinstance(seed_n, list):
                proj = self.in_proj[seed_n, ...].mean(dim=0, keepdim=True)
            else:
                proj = self.in_proj[seed_n:seed_n + 1, ...]
            proj = torch.cat([proj.to(float_type)] * batch_size, 0)
            out = ca_init.permute(0, 2, 3, 1).reshape(batch_size, self.ds_size * self.ds_size, self.n_filter)
            out = torch.bmm(out, proj).reshape(batch_size, self.ds_size, self.ds_size, self.n_filter).permute(0, 3, 1, 2).contiguous()

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        auto_reg_grads = []
        for c in range(self.n_calls):
            out_new = out
            if not self.auto_reg:
                out_new = self.frac_norm[0 if self.shared_params else c](out_new)
            if self.adain:
                s_fact, b_fact = torch.split(cond_factors[0 if self.shared_params else c], self.n_filter, dim=1)
                s_fact = s_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                b_fact = b_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                out_new = (s_fact * out_new) + b_fact
            elif self.dyncin:
                out_new = self.inj_cond[0 if self.shared_params else c](out_new, lat)
            else:
                c_fact = cond_factors[0 if self.shared_params else c].view(batch_size, self.n_filter, 1, 1).contiguous().repeat(1, 1, self.ds_size, self.ds_size)
                out_new = torch.cat([out_new, c_fact], dim=1)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out + (leak_factor * out_new)
            if self.auto_reg and out.requires_grad:
                with torch.no_grad():
                    auto_reg_grad = (2e-3 / out.numel()) * out
                auto_reg_grads.append(auto_reg_grad)
                out.register_hook(lambda grad: grad + auto_reg_grads.pop() if len(auto_reg_grads) > 0 else grad)
            out_embs.append(out)

        out = self.conv_img(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

