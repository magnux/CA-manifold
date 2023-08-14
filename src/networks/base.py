import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.residualblocks import ResidualBlockS
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.linearresidualblocks import LinearResidualBlockS
from src.layers.dynalinear import DynaLinear
from src.layers.irm import IRMLinear
from src.layers.augment.augment import AugmentPipe, augpipe_specs
from src.utils.loss_utils import vae_sample_gaussian, vae_gaussian_kl_loss
from src.layers.posencoding import sin_cos_pos_encoding_dyn
from src.layers.sequentialcond import SequentialCond
import numpy as np
from itertools import chain


class Classifier(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat):
        return self.labs(lat)


class Discriminator(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, auto_reg=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.embed_size = embed_size
        self.auto_reg = auto_reg

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_yembed = nn.Linear(n_labels, self.embed_size)
        self.lat_to_score = DynaLinear(self.embed_size, self.lat_size, 1)

    def forward(self, lat, y):
        assert (lat.size(0) == y.size(0))
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.labs_to_yembed(yembed)
        # yembed = F.normalize(yembed)
        score = self.lat_to_score(lat, yembed)

        if self.auto_reg and self.training and score.requires_grad:
            score.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

        return score


class Generator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, norm_z=False, auto_reg=False, irm_z=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.norm_z = norm_z
        self.auto_reg = auto_reg
        self.irm_z = irm_z
        self.register_buffer('embedding_mat', torch.eye(n_labels))

        if self.irm_z:
            self.irm_layer = IRMLinear(self.z_dim, n_layers=4)

        self.labs_to_yembed = nn.Linear(n_labels, self.embed_size)
        # nn.init.uniform_(self.labs_to_yembed.weight, 0, 1)
        # nn.init.constant_(self.labs_to_yembed.bias, 0.)
        self.yembed_to_lat = DynaLinear(self.z_dim, self.embed_size, self.lat_size)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        if self.norm_z:
            z = F.normalize(z)

        if self.irm_z:
            z = self.irm_layer(z)

        yembed = self.labs_to_yembed(yembed)
        # yembed = F.normalize(yembed)

        lat = self.yembed_to_lat(yembed, z)

        if self.auto_reg and self.training and lat.requires_grad:
            lat.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

        # lat = normalize_2nd_moment(lat)

        return lat


class MultiGenerator(nn.Module):
    def __init__(self, n_calls, n_labels, lat_size, z_dim, embed_size, image_size, norm_z=False, auto_reg=False, irm_z=False, **kwargs):
        super().__init__()
        self.n_calls = n_calls
        self.z_dim = z_dim
        self.lat_mult = int(np.log2(image_size))
        self.generators = nn.ModuleList(
            [Generator(n_labels, lat_size, z_dim, embed_size, norm_z, auto_reg, irm_z, **kwargs) for _ in range(self.lat_mult)]
        )

    def forward(self, z, y):
        lat = []
        for c in range(self.lat_mult):
            lat.append(self.generators[c](z[:, c * self.z_dim:(c + 1) * self.z_dim], y))
        return torch.cat(lat, 1)


class LabsEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, auto_reg=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.embed_size = embed_size
        self.auto_reg = auto_reg
        self.register_buffer('embedding_mat', torch.eye(n_labels))

        self.labs_to_yembed = nn.Linear(n_labels, self.embed_size)
        # nn.init.uniform_(self.labs_to_yembed.weight, 0, 1)
        # nn.init.constant_(self.labs_to_yembed.bias, 0.)
        self.yembed_to_lat = nn.Linear(self.embed_size, self.lat_size)

    def forward(self, y):
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.labs_to_yembed(yembed)
        # yembed = F.normalize(yembed)
        lat = self.yembed_to_lat(yembed)

        if self.auto_reg and self.training and lat.requires_grad:
            lat.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

        # lat = normalize_2nd_moment(lat)

        return lat


# class ZImgEncoder(nn.Module):
#     def __init__(self, z_dim, image_size, channels, multi_cut=True, **kwargs):
#         super().__init__()
#         self.z_dim = z_dim
#         self.image_size = image_size
#         self.in_chan = channels
#         self.multi_cut = multi_cut
#
#         self.split_sizes = [self.in_chan] * 7 if self.multi_cut else [self.in_chan]
#         self.conv_state_size = [self.in_chan, self.image_size, self.image_size, self.in_chan * self.image_size,
#                                 self.in_chan * self.image_size, self.image_size ** 2, 1] if self.multi_cut else [self.in_chan]
#
#         self.z_conv = nn.Conv2d(self.in_chan, sum(self.split_sizes), 1, 1, 0)
#         self.z_fc = nn.Linear(sum(self.conv_state_size), z_dim)
#
#     def forward(self, img):
#         batch_size = int(img.size(0))
#
#         conv_state = self.z_conv(img)
#         if self.multi_cut:
#             conv_state_f, conv_state_h, conv_state_w, conv_state_fh, conv_state_fw, conv_state_hw, conv_state_g = torch.split(conv_state, self.split_sizes, dim=1)
#             conv_state = torch.cat([conv_state_f.mean(dim=(2, 3)),
#                                     conv_state_h.mean(dim=(1, 3)),
#                                     conv_state_w.mean(dim=(1, 2)),
#                                     conv_state_fh.mean(dim=3).view(batch_size, -1),
#                                     conv_state_fw.mean(dim=2).view(batch_size, -1),
#                                     conv_state_hw.mean(dim=1).view(batch_size, -1),
#                                     conv_state_g.mean(dim=(1, 2, 3)).view(batch_size, 1)], dim=1)
#         else:
#             conv_state = conv_state.mean(dim=(2, 3))
#
#         z = self.z_fc(conv_state)
#
#         return z


# class LabsImgEncoder(nn.Module):
#     def __init__(self, n_labels, lat_size, z_dim, embed_size, image_size, channels, multi_cut=True, auto_reg=False,**kwargs):
#         super().__init__()
#         self.z_img_encoder = ZImgEncoder(z_dim, image_size, channels, multi_cut, **kwargs)
#         self.generator = Generator(n_labels, lat_size, z_dim, embed_size, False, False, False, **kwargs)
#
#     def forward(self, img, y):
#         z = self.z_img_encoder(img)
#         lat = self.generator(z, y)
#
#         return lat


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, lat_size, auto_reg=False, wavify_lat=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size * (2 if wavify_lat else 1)
        self.auto_reg = auto_reg

        self.lat_to_score = nn.Linear(self.lat_size, 1)

    def forward(self, lat):
        score = self.lat_to_score(lat)

        if self.auto_reg and self.training and score.requires_grad:
            score.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

        return score


class LongUnconditionalDiscriminator(UnconditionalDiscriminator):
    def __init__(self, lat_size, auto_reg=False, **kwargs):
        super().__init__(lat_size * 2, auto_reg, **kwargs)


class UnconditionalGenerator(nn.Module):
    def __init__(self, lat_size, z_dim, n_calls=4, norm_z=False, auto_reg=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.norm_z = norm_z
        self.n_calls = n_calls
        self.auto_reg = auto_reg

        self.z_to_lat = nn.Linear(self.z_dim, self.lat_size)

    def forward(self, z):
        if self.norm_z:
            z = F.normalize(z)

        lat = self.z_to_lat(z)

        if self.auto_reg and self.training and lat.requires_grad:
            lat.register_hook(lambda grad: grad + ((grad - 1e-4) / (grad - 1e-4).norm()))

        return lat


class VarDiscriminator(nn.Module):
    def __init__(self, lat_size, z_dim, norm_lat=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.norm_lat = norm_lat
        self.lat_to_z = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, z_dim * 2),
        )
        self.z_to_score = nn.Linear(z_dim, 1, bias=False)

    def forward(self, lat):
        z = self.lat_to_z(lat)
        z_mu, z_log_var = torch.split(z, self.z_dim, 1)
        z = vae_sample_gaussian(z_mu, z_log_var)
        score = self.z_to_score(z)
        loss_kl = vae_gaussian_kl_loss(z_mu, z_log_var)

        return score, loss_kl


class VarEncoder(nn.Module):
    def __init__(self, lat_size, z_dim, norm_lat=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.norm_lat = norm_lat
        self.lat_to_z = nn.Sequential(
            LinearResidualBlock(self.lat_size, self.lat_size),
            LinearResidualBlock(self.lat_size, z_dim * 2),
        )

    def forward(self, lat):
        z = self.lat_to_z(lat)
        z_mu, z_log_var = torch.split(z, self.z_dim, 1)
        z = vae_sample_gaussian(z_mu, z_log_var)
        loss_kl = vae_gaussian_kl_loss(z_mu, z_log_var)

        return z, loss_kl


class LetterEncoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.lat_to_letters = nn.Sequential(
            ResidualBlock(1, letter_channels * letter_bits, None, 1, 1, 0, 1, nn.Conv1d),
            *([ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, 1, nn.Conv1d) for _ in range(n_layers-1)]),
        )

    def forward(self, lat):
        lat = lat.view(lat.size(0), 1, self.lat_size)
        letters = self.lat_to_letters(lat)
        letters = letters.view(letters.size(0), self.letter_channels, self.letter_bits * self.lat_size)
        letters = F.softmax(letters, dim=1)

        return letters


class LetterDecoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.letters_to_lat = nn.Sequential(
            *([ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, 1, nn.Conv1d) for _ in range(n_layers-1)]),
            ResidualBlock(letter_channels * letter_bits, 1, None, 1, 1, 0, 1, nn.Conv1d),
        )

    def forward(self, letters):
        letters = letters.view(letters.size(0), self.letter_channels * self.letter_bits, self.lat_size)
        lat = self.letters_to_lat(letters)
        lat = lat.squeeze(dim=1)

        return lat


class IRMTranslator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.irm_layer = IRMLinear(self.lat_size, n_layers=4)

    def forward(self, lat):
        return self.irm_layer(lat)


class HBUnfolder(nn.Module):
    def __init__(self, lat_size, exp_factor=8, **kwargs):
        super().__init__()
        assert lat_size % 2 == 0
        self.lat_size = lat_size
        self.exp_factor = exp_factor
        self.lat_to_lat = nn.Linear(self.lat_size, self.lat_size * self.exp_factor)

    def forward(self, lat):
        lat = self.lat_to_lat(lat)
        u, v = torch.chunk(lat, 2, 1)
        lat = torch.cat([v * u.exp(), v * (-u).exp()], dim=1)
        return lat


class HBFolder(nn.Module):
    def __init__(self, lat_size, exp_factor=8, **kwargs):
        super().__init__()
        assert lat_size % 2 == 0
        self.lat_size = lat_size
        self.exp_factor = exp_factor
        self.lat_to_lat = nn.Linear(self.lat_size * self.exp_factor, self.lat_size)

    def forward(self, lat):
        lat = F.relu(self.lat_to_lat(lat)) + 1e-4  # We force the space to be positive
        x, y = torch.chunk(lat, 2, 1)
        lat = torch.cat([(x / y).pow(0.5).log(), (x * y).pow(0.5)], dim=1)
        return lat


class IRMUnfolder(nn.Module):
    def __init__(self, lat_size, exp_factor=8, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.exp_factor = exp_factor
        self.lat_to_lat = nn.ModuleList([
            IRMLinear(self.lat_size, n_layers=3) for _ in range(exp_factor)
        ])

    def forward(self, lat):
        new_lat = []
        for i in range(self.exp_factor):
            new_lat.append(self.lat_to_lat[i](lat))
        new_lat = torch.cat(new_lat, dim=1)
        return new_lat


class IRMFolder(nn.Module):
    def __init__(self, lat_size, exp_factor=8, **kwargs):
        super().__init__()
        assert lat_size % 2 == 0
        self.lat_size = lat_size
        self.exp_factor = exp_factor
        self.lat_to_lat = nn.Sequential(
            LinearResidualBlock(self.lat_size * exp_factor, self.lat_size * exp_factor // 2, self.lat_size * exp_factor // 2),
            LinearResidualBlock(self.lat_size * exp_factor // 2, self.lat_size, self.lat_size)
        )

    def forward(self, lat):
        return self.lat_to_lat(lat)


class LatEncoder(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size if lat_size > 3 else 512
        self.lat_to_lat = nn.Sequential(
            LinearResidualBlock(lat_size, self.lat_size),
            *([LinearResidualBlock(self.lat_size, self.lat_size) for _ in range(6)]),
            LinearResidualBlock(self.lat_size, lat_size),
        )

    def forward(self, lat):
        lat = self.lat_to_lat(lat)

        return lat


class LatCompressor(nn.Module):
    def __init__(self, lat_size, n_goals, **kwargs):
        super().__init__()
        self.lat_size = lat_size if lat_size > 3 else 512
        self.lat_to_lat = nn.Sequential(
            LinearResidualBlock(lat_size * n_goals, self.lat_size),
            LinearResidualBlock(self.lat_size, lat_size),
        )

    def forward(self, lats):
        lat = self.lat_to_lat(torch.cat(lats, dim=1))

        return lat


class ContrastiveCritic(nn.Module):
    def __init__(self, lat_size, norm_lat=True, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.norm_lat = norm_lat

        self.lat_to_lat = nn.Sequential(
            *([LinearResidualBlockS(self.lat_size, self.lat_size) for _ in range(n_layers)]),
        )
        self.lat_to_score = nn.Linear(self.lat_size * 2 + self.lat_size ** 2, 1)

    def forward(self, lat_a, lat_b):

        lat_a = self.lat_to_lat(lat_a)
        lat_b = self.lat_to_lat(lat_b)
        lat_mat = ((lat_a[:, :, None] * lat_b[:, None, :]) / self.lat_size).view(lat_a.size(0), lat_a.size(1) ** 2)
        lat_mat = torch.cat([lat_a, lat_b, lat_mat], dim=1)
        score = self.lat_to_score(lat_mat)
        return score


class ContrastiveMixer(nn.Module):
    def __init__(self, lat_size, norm_lat=True, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.norm_lat = norm_lat

        self.lat_to_mix = nn.Linear(self.lat_size ** 2, int(self.lat_size ** 0.5))

        self.mix_to_lat = nn.Sequential(
            LinearResidualBlockS(self.lat_size * 2 + int(self.lat_size ** 0.5), self.lat_size),
            *([LinearResidualBlockS(self.lat_size, self.lat_size) for _ in range(n_layers - 1)]),
        )

    def forward(self, lat_a, lat_b):

        lat_mat = ((lat_a[:, :, None] * lat_b[:, None, :]) / self.lat_size).view(lat_a.size(0), lat_a.size(1) ** 2)

        mix = self.lat_to_mix(lat_mat)
        mix = torch.cat([lat_a, lat_b, mix], dim=1)
        lat = self.mix_to_lat(mix)

        return lat


class EmbSampler(nn.Module):
    def __init__(self, lat_size, n_calls, lat_channels=4, n_layers=4,  **kwargs):
        super().__init__()
        self.lat_size = lat_size if lat_size > 3 else 512
        self.n_calls = n_calls
        self.lat_channels = lat_channels
        self.labs_encoder = LabsEncoder(lat_size=lat_size, **kwargs)

        self.lat_to_lat = nn.Sequential(
            LinearResidualBlockS(lat_size * 2, lat_size * 2),
            *([LinearResidualBlockS(lat_size * 2, lat_size * 2) for _ in range(n_layers - 2)]),
            LinearResidualBlockS(lat_size * 2, lat_size),
        )

    def forward(self, lat_init, labels):
        lat_out = self.lat_to_lat(torch.cat([lat_init, self.labs_encoder(labels)], 1))

        return lat_out


class EasyAugmentPipe(AugmentPipe):
    def __init__(self, spec_name='bgc', **kwargs):
        super().__init__(**augpipe_specs[spec_name])


class LetterEncoderS(nn.Module):
    def __init__(self, lat_size, letter_channels=2, letter_bits=8, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.lat_to_letters = nn.Sequential(
            ResidualBlockS(1, letter_channels * letter_bits, dim=1, pos_enc=True, image_size=self.lat_size, attention=True, attn_patches=self.lat_size//letter_bits),
            *([ResidualBlockS(letter_channels * letter_bits, letter_channels * letter_bits, dim=1, pos_enc=True, image_size=self.lat_size, attention=True, attn_patches=self.lat_size//letter_bits) for _ in range(n_layers-1)]),
        )

    def forward(self, lat):
        lat = lat.view(lat.size(0), 1, self.lat_size)
        letters = self.lat_to_letters(lat)
        letters = letters.view(letters.size(0), self.letter_channels, self.letter_bits * self.lat_size)
        letters = F.softmax(letters, dim=1)

        return letters


class LetterDecoderS(nn.Module):
    def __init__(self, lat_size, letter_channels=2, letter_bits=8, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.letters_to_lat = nn.Sequential(
            *([ResidualBlockS(letter_channels * letter_bits, letter_channels * letter_bits, dim=1, pos_enc=True, image_size=self.lat_size, attention=True, attn_patches=self.lat_size//letter_bits) for _ in range(n_layers-1)]),
            ResidualBlockS(letter_channels * letter_bits, 1, dim=1, pos_enc=True, image_size=self.lat_size, attention=True, attn_patches=self.lat_size//letter_bits),
        )

    def forward(self, letters):
        letters = letters.view(letters.size(0), self.letter_channels * self.letter_bits, self.lat_size)
        lat = self.letters_to_lat(letters)
        lat = lat.view(letters.size(0), self.lat_size)

        return lat
