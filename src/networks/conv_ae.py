import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.imagescaling import DownScale, UpScale
from src.layers.lambd import LambdaLayer
from src.layers.dynaresidualblock import DynaResidualBlock
from src.networks.base import LabsEncoder
from src.utils.model_utils import ca_seed
from src.utils.loss_utils import sample_from_discretized_mix_logistic


class Encoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, n_calls, shared_params,
                 injected=False, adain=False, dyncin=False, multi_cut=True, **kwargs):
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
                    LinearResidualBlock(self.lat_size, self.n_filter * 2 * (1 if self.shared_params else self.n_calls)),
                )
            elif self.dyncin:
                self.inj_cond = DynaResidualBlock(self.lat_size, self.n_filter, self.n_filter, self.n_filter)
            else:
                self.inj_cond = ResidualBlock(self.n_filter + self.lat_size, self.n_filter, None, 1, 1, 0)

        self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        self.frac_conv = nn.ModuleList([nn.Sequential(
                                            ResidualBlock(self.n_filter, self.n_filter, self.n_filter * 4, 1, 1, 0),
                                            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1)
                                        ) for _ in range(1 if self.shared_params else self.n_calls)])

        self.out_conv = ResidualBlock(self.n_filter, sum(self.split_sizes), None, 1, 1, 0)
        self.out_to_lat = nn.Sequential(
            LinearResidualBlock(sum(self.conv_state_size), self.lat_size, self.lat_size * 2),
            LinearResidualBlock(self.lat_size, self.lat_size),
            nn.Linear(self.lat_size, self.lat_size)
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
                inj_lat = inj_lat.view(batch_size, self.lat_size, 1, 1).repeat(1, 1, self.ds_size, self.ds_size)
                out = self.inj_cond(torch.cat([out, inj_lat], dim=1))

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = self.frac_norm[0 if self.shared_params else c](out)
            if self.injected:
                if self.adain:
                    s_fact, b_fact = torch.split(cond_factors[0 if self.shared_params else c], self.n_filter, dim=1)
                    s_fact = s_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                    b_fact = b_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                    out_new = (s_fact * out_new) + b_fact
                elif self.dyncin:
                    out_new = self.inj_cond(out_new, inj_lat)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out + (leak_factor * out_new)
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
                 adain=False, dyncin=False, log_mix_out=False, **kwargs):
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
        self.merge_sizes = [self.n_filter, self.n_filter, self.n_filter, 1, self.n_filter]
        self.conv_state_size = [self.n_filter, self.n_filter * self.ds_size, self.n_filter * self.ds_size, self.ds_size ** 2]

        self.frac_norm = nn.ModuleList([nn.InstanceNorm2d(self.n_filter) for _ in range(1 if self.shared_params else self.n_calls)])
        self.frac_conv = nn.ModuleList([nn.Sequential(
                                            ResidualBlock(self.n_filter, self.n_filter, self.n_filter * 4, 1, 1, 0),
                                            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1)
                                        ) for _ in range(1 if self.shared_params else self.n_calls)])

        if self.adain:
            self.lat_to_facts = nn.Sequential(
                LinearResidualBlock(self.lat_size, self.lat_size),
                LinearResidualBlock(self.lat_size, self.n_filter * 2 * (1 if self.shared_params else self.n_calls), self.lat_size * 2),
            )
        elif self.dyncin:
            self.dyn_conv = DynaResidualBlock(self.lat_size, self.n_filter, self.n_filter, self.n_filter)
        else:
            self.lat_to_out = nn.Sequential(
                LinearResidualBlock(self.lat_size, self.lat_size),
                LinearResidualBlock(self.lat_size, sum(self.conv_state_size), self.lat_size * 2),
            )

        self.in_norm = nn.InstanceNorm2d(self.n_filter)
        self.in_conv = ResidualBlock(self.n_filter if self.adain or self.dyncin else sum(self.merge_sizes), self.n_filter, None, 1, 1, 0)

        self.conv_img = nn.Sequential(
            ResidualBlock(self.n_filter, self.n_filter, None, 3, 1, 1),
            # *([LambdaLayer(lambda x: F.interpolate(x, size=self.image_size))] if self.ds_size < self.image_size else []),
            *([UpScale(self.n_filter, self.n_filter, self.ds_size, self.image_size)] if self.ds_size < self.image_size else []),
            nn.Conv2d(self.n_filter, 10 * ((self.out_chan * 3) + 1) if self.log_mix_out else self.out_chan, 3, 1, 1),
        )

    def forward(self, lat, ca_init=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32

        if self.adain or self.dyncin:
            if self.adain:
                cond_factors = self.lat_to_facts(lat)
                cond_factors = torch.split(cond_factors, self.n_filter * 2, dim=1)
            if ca_init is None:
                out = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
            else:
                out = self.in_norm(ca_init)
                out = self.in_conv(out)
        else:
            conv_state = self.lat_to_out(lat)
            cs_f_m, cs_fh, cs_fw, cs_hw = torch.split(conv_state, self.conv_state_size, dim=1)
            cs_f_m = cs_f_m.view(batch_size, self.n_filter, 1, 1).repeat(1, 1, self.ds_size, self.ds_size)
            cs_fh = cs_fh.view(batch_size, self.n_filter, self.ds_size, 1).repeat(1, 1, 1, self.ds_size)
            cs_fw = cs_fw.view(batch_size, self.n_filter, 1, self.ds_size).repeat(1, 1, self.ds_size, 1)
            cs_hw = cs_hw.view(batch_size, 1, self.ds_size, self.ds_size)
            if ca_init is None:
                ca_init = ca_seed(batch_size, self.n_filter, self.ds_size, lat.device).to(float_type)
            else:
                ca_init = self.in_norm(ca_init)
            out = torch.cat([cs_f_m, cs_fh, cs_fw, cs_hw, ca_init], dim=1)
            out = self.in_conv(out)

        out_embs = [out]
        leak_factor = torch.clamp(self.leak_factor, 1e-3, 1e3)
        for c in range(self.n_calls):
            out_new = self.frac_norm[0 if self.shared_params else c](out)
            if self.adain:
                s_fact, b_fact = torch.split(cond_factors[0 if self.shared_params else c], self.n_filter, dim=1)
                s_fact = s_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                b_fact = b_fact.view(batch_size, self.n_filter, 1, 1).contiguous()
                out_new = (s_fact * out_new) + b_fact
            elif self.dyncin:
                out_new = self.dyn_conv(out_new, lat)
            out_new = self.frac_conv[0 if self.shared_params else c](out_new)
            out = out + (leak_factor * out_new)
            out_embs.append(out)

        out = self.conv_img(out)
        out_raw = out
        if self.log_mix_out:
            out = sample_from_discretized_mix_logistic(out, 10)
        else:
            out = out.clamp(-1., 1.)

        return out, out_embs, out_raw

