import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.data
import torch.utils.data.distributed
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.sobel import SinSobel
from src.layers.mixresidualblock import MixResidualBlock
from src.layers.dynaresidualblock import DynaResidualBlock
from src.layers.complexwave import complex_wave
from src.layers.lambd import LambdaLayer
from src.networks.base import LabsEncoder
from src.utils.loss_utils import sample_from_discretized_mix_logistic
from src.layers.posencoding import PosEncoding
import numpy as np


class InjectedEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, image_size, ds_size, channels, n_filter, z_out=False, z_dim=0, auto_reg=False, zoom_factor=0, **kwargs):
        super().__init__()
        self.injected = True
        self.n_labels = n_labels
        self.image_size = image_size
        self.ds_size = ds_size
        self.in_chan = channels
        self.n_filter = n_filter
        self.lat_size = lat_size
        self.auto_reg = auto_reg
        self.zoom_factor = zoom_factor

        self.n_layers = int(np.log2(image_size)) - 1 + zoom_factor

        self.conv_img = nn.Conv2d(self.in_chan, self.n_filter, 1, 1, 0)
        self.layer_grad = nn.ModuleList([SinSobel(self.n_filter * (l + 1), 3, 1) for l in range(self.n_layers)])
        self.layer_conv = nn.ModuleList([MixResidualBlock(self.lat_size, self.n_filter * (l + 1) * 3, self.n_filter * (l + 2), self.n_filter * (l + 1)) for l in range(self.n_layers)])

        self.out_to_lat = LinearResidualBlock(self.n_filter * (self.n_layers + 1) * 4, self.lat_size if not z_out else z_dim)

    def forward(self, x, inj_lat=None):
        assert (inj_lat is not None) == self.injected, 'latent should only be passed to injected encoders'
        batch_size = x.size(0)

        out = x
        if self.zoom_factor > 0:
            out = F.interpolate(out, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
        out = self.conv_img(out)

        out_embs = [out]
        for l in range(self.n_layers):
            out = self.layer_grad[l](out)
            out = self.layer_conv[l](out, inj_lat)
            out = F.interpolate(out, scale_factor=0.5, mode='bilinear', align_corners=False)

            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))
            out_embs.append(out)

        lat = self.out_to_lat(out.reshape(batch_size, self.n_filter * (self.n_layers + 1) * 4))

        return lat, out_embs, None


class LabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = LabsEncoder(**kwargs)

    def forward(self, x, labels):
        self.inj_lat = self.labs_encoder(labels)
        # if g_factor > 0.:
        #     self.inj_lat = (1. - g_factor) * self.inj_lat + g_factor * torch.randn_like(self.inj_lat)
        return super().forward(x, self.inj_lat)

    @property
    def lr_mul(self):
        return self.labs_encoder.yembed_to_lat.lr_mul

    @lr_mul.setter
    def lr_mul(self, lr_mul):
        self.labs_encoder.yembed_to_lat.lr_mul = lr_mul


class MultiLabsInjectedEncoder(InjectedEncoder):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.labs_encoder = nn.ModuleList(
            [LabsEncoder(**kwargs) for _ in range(self.lat_mult)]
        )

    def forward(self, x, labels):
        inj_lat = []
        for l in range(self.lat_mult):
            inj_lat.append(self.labs_encoder[l](labels))
        self.inj_lat = torch.cat(inj_lat, 1)
        # if g_factor > 0.:
        #     self.inj_lat = (1. - g_factor) * self.inj_lat + g_factor * torch.randn_like(self.inj_lat)
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
                 skip_fire=False, log_mix_out=False, causal=False, gated=True, env_feedback=False, multi_cut=True, auto_reg=False, ce_out=False, zoom_factor=0, **kwargs):
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
        self.zoom_factor = zoom_factor

        self.n_layers = int(np.log2(image_size)) - 1 + zoom_factor

        self.cell_grads = SinSobel(self.n_filter, 3, 1)

        # self.register_buffer('noise_seed', torch.randn(1, self.out_chan, self.image_size, self.image_size).clamp_(-1, 1))

        self.in_conv = nn.Conv2d(self.out_chan, self.n_filter, 1, 1, 0)

        self.wave_inits = nn.ParameterList([
            nn.Parameter(torch.nn.init.orthogonal_(torch.empty(1, self.n_filter * max(l, 1) * 2, 1, 1)).repeat(1, 1, (2 ** zoom_factor * self.image_size) // 2 ** (l - 1), (2 ** zoom_factor * self.image_size) // 2 ** (l - 1)))
        for l in range(self.n_layers + 1, 0, -1)])

        self.wave_grads = nn.ModuleList([
            SinSobel(self.n_filter * max(l, 1) * (2 if l < self.n_layers + 1 else 1), 3, 1)
        for l in range(self.n_layers + 1, 0, -1)])

        self.wave_convs = nn.ModuleList([
            ResidualBlock(self.n_filter * max(l, 1) * (2 if l < self.n_layers + 1 else 1) * 3, self.n_filter * max(l - 1, 1), self.n_filter * max(l, 1))
        for l in range(self.n_layers + 1, 0, -1)])

        self.wave_convs_dyn = nn.ModuleList([
            DynaResidualBlock(self.lat_size, self.n_filter * max(l, 1) * (2 if l < self.n_layers + 1 else 1) * 3, self.n_filter * max(l - 1, 1), self.n_filter * max(l, 1), lat_factor=(self.lat_size ** 0.9) / self.lat_size)
        for l in range(self.n_layers + 1, 0, -1)])

        self.cell_to_cell = ResidualBlock(self.n_filter * (self.cell_grads.c_factor + 4), self.n_filter * (2 if self.gated else 1) // 2, self.n_filter * 2, 1, 1, 0)

        self.call_enc = PosEncoding(self.n_calls, 0, 4, self.n_calls)
        self.cell_to_cell_dyn = DynaResidualBlock(self.lat_size + self.call_enc.size(), self.n_filter * (self.cell_grads.c_factor + 4), self.n_filter * (2 if self.gated else 1) // 2, self.n_filter * 2)

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

    def noise_factor(self, c):
        return (0.5 * (np.cos(np.pi * c / self.n_calls) + 1)) ** 0.69

    def forward(self, lat, img_init=None, noise_inits=None, call_init=None, call_end=None):
        batch_size = lat.size(0)
        float_type = torch.float16 if isinstance(lat, torch.cuda.HalfTensor) else torch.float32
        call_init = call_init if call_init is not None else 0
        call_end = call_end if call_end is not None else self.n_calls
        if noise_inits is None:
            noise_inits = [torch.randn(batch_size, self.out_chan, self.image_size, self.image_size, device=lat.device) for _ in range(call_end - call_init)]

        waves = []
        for l, (wave_init, wave_grad, wave_conv, wave_conv_dyn) in enumerate(zip(self.wave_inits, self.wave_grads, self.wave_convs, self.wave_convs_dyn)):
            wave_init, wave_init_dyn = torch.chunk(wave_init, 2, 1)
            wave_init = wave_init.repeat(batch_size, 1, 1, 1)
            wave_init_dyn = wave_init_dyn.repeat(batch_size, 1, 1, 1)
            if l > 0:
                wave = torch.cat([wave, wave_init], 1)
                wave_dyn = torch.cat([wave_dyn, wave_init_dyn], 1)
            else:
                wave = wave_init
                wave_dyn = wave_init_dyn
            wave = wave_grad(wave)
            wave = wave_conv(wave)
            wave_dyn = wave_grad(wave_dyn)
            wave_dyn = wave_conv_dyn(wave_dyn, lat)
            if l < self.n_layers:
                wave = F.interpolate(wave, scale_factor=2, mode='bilinear', align_corners=False)
                wave_dyn = F.interpolate(wave_dyn, scale_factor=2, mode='bilinear', align_corners=False)
            elif wave.shape[2] > (2 ** self.zoom_factor * self.image_size):
                wave = F.interpolate(wave, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
                wave_dyn = F.interpolate(wave_dyn, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
            waves.append(torch.cat([wave, wave_dyn], dim=1))

        wave = complex_wave(waves[-1])

        if img_init is not None:
            out = img_init
        else:
            out = torch.randn(batch_size, self.out_chan, self.image_size, self.image_size, device=lat.device).clamp_(-1, 1).to(float_type)

        out_embs = [out]
        noise_outs = []
        call_range = range(call_init, call_end)
        for c in call_range:
            if self.causal:
                out = F.pad(out, [0, 1, 0, 1])

            out = out + self.noise_factor(c) * noise_inits[c - call_init]

            cell_in = out

            if self.zoom_factor > 0:
                cell_in = F.interpolate(cell_in, size=(2 ** self.zoom_factor * self.image_size), mode='bilinear', align_corners=False)
            cell_in = self.in_conv(cell_in)

            cell_in = self.cell_grads(cell_in)

            cell_in = torch.cat([cell_in, wave], dim=1)

            cell_out = self.cell_to_cell(cell_in)

            call_lat = self.call_enc(lat, c)
            cell_out_dyn = self.cell_to_cell_dyn(cell_in, call_lat)

            if self.gated:
                cell_out, cell_out_gate = torch.chunk(cell_out, 2, dim=1)
                cell_out = cell_out * F.relu(cell_out_gate + 1)
                cell_out_dyn, cell_out_dyn_gate = torch.chunk(cell_out_dyn, 2, dim=1)
                cell_out_dyn = cell_out_dyn * F.relu(cell_out_dyn_gate + 1)

            cell_out = torch.cat([cell_out, cell_out_dyn], dim=1)

            if self.zoom_factor > 0:
                cell_out = F.interpolate(cell_out, size=self.image_size, mode='bilinear', align_corners=False)

            noise_out = self.out_conv(cell_out)
            noise_outs.append(noise_out)

            out = out - self.noise_factor(c) * noise_out

            if self.fire_rate < 1.0:
                out = out * (torch.rand([batch_size, 1, out.size(2), out.size(3)], device=lat.device) <= self.fire_rate).to(float_type)

            if self.skip_fire:
                if c % 2 == 0:
                    out = out * self.skip_fire_mask.to(device=lat.device).to(float_type)
                else:
                    out = out * (1 - self.skip_fire_mask.to(device=lat.device).to(float_type))

            if self.causal:
                out = out[:, :, 1:, 1:]

            if self.auto_reg and out.requires_grad:
                out.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

            out_embs.append(out)

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

        return out, out_embs, noise_outs

    def mix_forward(self, lat_a, lat_b, ratio):
        batch_size = lat_a.size(0)
        float_type = torch.float16 if isinstance(lat_a, torch.cuda.HalfTensor) else torch.float32
        image_gen = torch.randn(batch_size, self.out_chan, self.image_size, self.image_size, device=lat_a.device).clamp_(-1, 1).to(float_type)
        for c in range(self.n_calls):
            noise_init = [torch.randn_like(image_gen).clamp_(-1, 1)]
            noise_out_a = self.forward(lat_a, image_gen, noise_init, c, c + 1)[2][0]
            noise_out_b = self.forward(lat_b, image_gen, noise_init, c, c + 1)[2][0]
            noise_out = ratio * noise_out_a + (1 - ratio) * noise_out_b
            image_gen = image_gen - self.noise_factor(c) * noise_out

        return image_gen
