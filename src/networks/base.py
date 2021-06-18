import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.irm import IRMLinear
from src.layers.augment.augment import AugmentPipe, augpipe_specs
from src.utils.loss_utils import vae_sample_gaussian, vae_gaussian_kl_loss


class Classifier(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat):
        return self.labs(lat)


class Discriminator(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.embed_size = embed_size

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.lat_to_score = nn.Linear(self.lat_size, n_labels, bias=False)

    def forward(self, lat, y):
        assert(lat.size(0) == y.size(0))
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        score = (self.lat_to_score(lat) * yembed).sum(dim=1, keepdim=True) * (1 / self.n_labels ** 0.5)

        return score


class Generator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, norm_z=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.norm_z = norm_z

        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.yembed_irm = nn.Sequential(
            nn.Linear(n_labels, self.embed_size),
            IRMLinear(self.embed_size, 2)
        )
        self.z_irm = nn.Sequential(
            IRMLinear(self.z_dim, 2),
            nn.Linear(self.z_dim, self.lat_size, bias=False),
        )
        self.lat_proj = nn.Linear(self.embed_size, 1, bias=False)

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        batch_size = z.size(0)

        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        if self.norm_z:
            z = F.normalize(z, dim=1)

        yembed = self.yembed_irm(yembed).view(batch_size, 1, self.embed_size)
        z = self.z_irm(z).view(batch_size, self.lat_size, 1)
        lat = torch.bmm(z, yembed)
        lat = self.lat_proj(lat).squeeze(2)

        return lat


class LabsEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))

        self.yembed_irm = nn.Sequential(
            nn.Linear(n_labels, self.embed_size),
            IRMLinear(self.embed_size, 2),
        )
        self.yembed_to_lat = nn.Linear(self.embed_size, lat_size, bias=False)

    def forward(self, y):
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.yembed_irm(yembed)
        lat = self.yembed_to_lat(yembed)

        return lat


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.lat_to_score = nn.Linear(self.lat_size, 1, bias=False)

    def forward(self, lat):
        score = self.lat_to_score(lat)

        return score


class UnconditionalGenerator(nn.Module):
    def __init__(self, lat_size, z_dim, norm_z=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.norm_z = norm_z

        self.z_irm = IRMLinear(self.z_dim, 3)
        self.z_to_lat = nn.Linear(self.z_dim, self.lat_size, bias=False)

    def forward(self, z):
        if self.norm_z:
            z = F.normalize(z, dim=1)

        z = self.z_irm(z)
        lat = self.z_to_lat(z)

        return lat


class VarDiscriminator(nn.Module):
    def __init__(self, lat_size, z_dim, norm_lat=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.norm_lat = norm_lat
        self.lat_to_z = nn.Linear(self.lat_size, z_dim * 2, bias=False)
        self.z_to_score = nn.Linear(z_dim, 1, bias=False)

    def forward(self, lat):
        z = self.lat_to_z(lat)
        z_mu, z_log_var = torch.split(z, self.z_dim, 1)
        z = vae_sample_gaussian(z_mu, z_log_var)
        score = self.z_to_score(z)
        kl_loss = vae_gaussian_kl_loss(z_mu, z_log_var)

        return score, kl_loss


class VarEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)
        self.lat_to_z = nn.Sequential(
            LinearResidualBlock(self.lat_size + embed_size, self.lat_size),
            LinearResidualBlock(self.lat_size, z_dim * 2),
        )

    def forward(self, lat, y):
        assert (lat.size(0) == y.size(0))
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        z = self.lat_to_z(torch.cat([lat, yembed], dim=1))
        z_mu, z_log_var = torch.split(z, self.z_dim, dim=1)

        return z_mu, z_log_var


class VarDecoder(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, embed_size)
        self.z_to_lat = nn.Sequential(
            LinearResidualBlock(z_dim + embed_size, self.lat_size),
            LinearResidualBlock(self.lat_size, self.lat_size),
        )

    def forward(self, z, y):
        assert (z.size(0) == y.size(0))
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)
        yembed = F.normalize(yembed)
        lat = self.z_to_lat(torch.cat([z, yembed], dim=1))

        return lat


class LetterEncoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, n_layers=4, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.lat_to_letters = nn.Sequential(
            ResidualBlock(1, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            *([ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d) for _ in range(n_layers-1)]),
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
            *([ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d) for _ in range(n_layers-1)]),
            ResidualBlock(letter_channels * letter_bits, 1, None, 1, 1, 0, nn.Conv1d),
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
        self.irm_layer = IRMLinear(self.lat_size, 4)

    def forward(self, lat):
        return self.irm_layer(lat)


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


class EasyAugmentPipe(AugmentPipe):
    def __init__(self, spec_name='bgc', **kwargs):
        super().__init__(**augpipe_specs[spec_name])
