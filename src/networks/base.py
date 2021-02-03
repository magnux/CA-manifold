import torch
import torch.nn as nn
import torch.nn.functional as F
from src.layers.residualblock import ResidualBlock
from src.layers.linearresidualblock import LinearResidualBlock
from src.layers.irm import IRMLinear


class Classifier(nn.Module):
    def __init__(self, n_labels, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, n_labels)

    def forward(self, lat):
        return self.labs(lat)


class Discriminator(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, norm_lat=False, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_proj = nn.Sequential(
            LinearResidualBlock(n_labels, self.embed_size, int(self.embed_size ** 0.5)),
            LinearResidualBlock(self.embed_size, (self.lat_size * 1) + 1, int(self.embed_size ** 0.5)),
        )
        self.norm_lat = norm_lat

    def forward(self, lat, y):
        assert(lat.size(0) == y.size(0))
        batch_size = lat.size(0)

        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        lat_proj = self.labs_to_proj(yembed)
        lat_proj, lat_bias = torch.split(lat_proj, [self.lat_size, 1], dim=1)
        lat_proj = lat_proj.view(batch_size, self.lat_size, 1)

        if self.norm_lat:
            lat = F.normalize(lat, dim=1)

        lat = lat.view(batch_size, 1, self.lat_size)
        score = torch.bmm(lat, lat_proj).squeeze(1) + lat_bias

        return score


class Generator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, norm_z=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_proj = nn.Sequential(
            LinearResidualBlock(n_labels, self.embed_size, int(self.embed_size ** 0.5)),
            LinearResidualBlock(self.embed_size, (self.z_dim * self.lat_size) + self.lat_size, int(self.embed_size ** 0.5)),
        )
        self.norm_z = norm_z

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

        lat_proj = self.labs_to_proj(yembed)
        lat_proj, lat_bias = torch.split(lat_proj, [self.z_dim * self.lat_size, self.lat_size], dim=1)
        lat_proj = lat_proj.view(batch_size, self.z_dim, self.lat_size)

        if self.norm_z:
            z = F.normalize(z, dim=1)
        else:
            z = z.clamp(-3, 3)

        z = z.view(batch_size, 1, self.z_dim)
        lat = torch.bmm(z, lat_proj).squeeze(1) + lat_bias

        return lat


class LabsEncoder(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.embedding_fc = nn.Linear(n_labels, lat_size, bias=False)

    def forward(self, y):
        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        yembed = self.embedding_fc(yembed)

        return yembed


class UnconditionalDiscriminator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.labs = nn.Linear(self.lat_size, 1, bias=False)

    def forward(self, lat):
        labs = self.labs(lat)

        return labs


class UnconditionalGenerator(nn.Module):
    def __init__(self, lat_size, z_dim, norm_z=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.z_dim = z_dim
        self.z_to_lat = nn.Linear(self.z_dim, self.lat_size, bias=False)
        self.norm_z = norm_z

    def forward(self, z):
        if self.norm_z:
            z = F.normalize(z, dim=1)
        else:
            z = z.clamp(-3, 3)

        lat = self.z_to_lat(z)

        return lat


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
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.lat_to_letters = nn.Sequential(
            ResidualBlock(1, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
        )

    def forward(self, lat):
        lat = lat.view(lat.size(0), 1, self.lat_size)
        letters = self.lat_to_letters(lat)
        letters = letters.view(letters.size(0), self.letter_channels, self.letter_bits * self.lat_size)
        letters = F.softmax(letters, dim=1)

        return letters


class LetterDecoder(nn.Module):
    def __init__(self, lat_size, letter_channels=4, letter_bits=16, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.letter_channels = letter_channels
        self.letter_bits = letter_bits
        self.letters_to_lat = nn.Sequential(
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, letter_channels * letter_bits, None, 1, 1, 0, nn.Conv1d),
            ResidualBlock(letter_channels * letter_bits, 1, None, 1, 1, 0, nn.Conv1d),
        )

    def forward(self, letters):
        letters = letters.view(letters.size(0), self.letter_channels * self.letter_bits, self.lat_size)
        lat = self.letters_to_lat(letters)
        lat = lat.squeeze(dim=1)

        return lat


class UnconditionalIRMTranslator(nn.Module):
    def __init__(self, lat_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.irm_layer = nn.Sequential(
            nn.Linear(self.lat_size, self.fhidden),
            IRMLinear(self.fhidden),
        )
        self.lat_out = nn.Linear(self.fhidden, self.lat_size)

    def forward(self, lat):
        lat = self.irm_layer(lat)
        lat = self.lat_out(lat)

        return lat


class IRMTranslator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_proj = nn.Sequential(
            LinearResidualBlock(n_labels, self.embed_size),
            LinearResidualBlock(self.embed_size, self.z_dim * self.lat_size, int(self.embed_size ** 0.5)),
        )
        self.irm_layer = nn.Sequential(
            nn.Linear(self.lat_size, self.fhidden),
            IRMLinear(self.fhidden),
            nn.Linear(self.fhidden, self.z_dim),
        )

    def forward(self, lat, y):
        assert (lat.size(0) == y.size(0))
        batch_size = lat.size(0)

        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        lat_proj = self.labs_to_proj(yembed)
        lat_proj = lat_proj.view(batch_size, self.z_dim, self.lat_size)

        lat = self.irm_layer(lat).view(batch_size, 1, self.z_dim)
        lat = torch.bmm(lat, lat_proj).squeeze(1)

        return lat


class IRMGenerator(nn.Module):
    def __init__(self, n_labels, lat_size, z_dim, embed_size, norm_z=True, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.z_dim = z_dim
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_proj = nn.Sequential(
            LinearResidualBlock(n_labels, self.embed_size),
            LinearResidualBlock(self.embed_size, self.z_dim * self.lat_size, int(self.embed_size ** 0.5)),
        )
        self.irm_layer = nn.Sequential(
            nn.Linear(self.z_dim, self.fhidden),
            IRMLinear(self.fhidden),
            nn.Linear(self.fhidden, self.z_dim),
        )
        self.norm_z = norm_z

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

        lat_proj = self.labs_to_proj(yembed)
        lat_proj = lat_proj.view(batch_size, self.z_dim, self.lat_size)

        if self.norm_z:
            z = F.normalize(z, dim=1)
        else:
            z = z.clamp(-3, 3)
        lat = self.irm_layer(z).view(batch_size, 1, self.z_dim)
        lat = torch.bmm(lat, lat_proj).squeeze(1)

        return lat


class IRMDiscriminator(nn.Module):
    def __init__(self, n_labels, lat_size, embed_size, **kwargs):
        super().__init__()
        self.lat_size = lat_size
        self.fhidden = lat_size if lat_size > 3 else 512
        self.embed_size = embed_size
        self.register_buffer('embedding_mat', torch.eye(n_labels))
        self.labs_to_proj = nn.Sequential(
            LinearResidualBlock(n_labels, self.embed_size),
            LinearResidualBlock(self.embed_size, self.lat_size, int(self.embed_size ** 0.5)),
        )
        self.irm_layer = IRMLinear(self.fhidden)

    def forward(self, lat, y):
        assert(lat.size(0) == y.size(0))
        batch_size = lat.size(0)

        if y.dtype is torch.int64:
            if y.dim() == 1:
                yembed = self.embedding_mat[y]
            else:
                yembed = y.to(torch.float32)
        else:
            yembed = y

        lat_proj = self.labs_to_proj(yembed)
        lat_proj = lat_proj.view(batch_size, self.lat_size, 1)
        lat = self.irm_layer(lat).view(batch_size, 1, self.lat_size)
        score = torch.bmm(lat, lat_proj).squeeze(1)

        return score


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
