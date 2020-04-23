import torch
import torch.nn as nn
import torch.nn.functional as F


class Centroids(nn.Module):
    def __init__(self, n_features, n_centroids, decay=0.99, eps=1e-5):
        super(Centroids, self).__init__()
        # Attributes
        self.n_features = n_features
        self.n_centroids = n_centroids
        self.decay = decay
        self.eps = eps

        centroids = torch.randn([n_features, n_centroids])
        self.register_buffer('centroids', centroids)
        self.register_buffer('cluster_size', torch.zeros(n_centroids))
        self.register_buffer('centroids_avg', centroids.clone())
        self.register_buffer('onehot_mat', torch.eye(self.n_centroids))

    def forward(self, x):

        x_flat = x
        if x.dim() > 2:
            x_flat = x_flat.transpose(-1, 1).contiguous()
            x_flat_size = x_flat.size()
            x_flat = x_flat.view(-1, self.n_features)

        dist = x_flat.pow(2).sum(1, keepdim=True) - 2 * x_flat @ self.centroids + self.centroids.pow(2).sum(0, keepdim=True)
        _, centroids_ind = (-dist).max(1)

        if self.training:
            centroids_onehot = self.onehot_mat[centroids_ind].type(x_flat.dtype)

            self.cluster_size.data.mul_(self.decay).add_(
                1 - self.decay, centroids_onehot.sum(0)
            )
            centroids_sum = x_flat.transpose(0, 1) @ centroids_onehot
            self.centroids_avg.data.mul_(self.decay).add_(1 - self.decay, centroids_sum)
            n = self.cluster_size.sum()
            cluster_size = (
                    (self.cluster_size + self.eps) / (n + self.n_centroids * self.eps) * n
            )
            centroids_normalized = self.centroids_avg / cluster_size.unsqueeze(0)
            self.centroids.data.copy_(centroids_normalized)

        quantize = F.embedding(centroids_ind, self.centroids.transpose(0, 1))
        if x.dim() > 2:
            quantize = quantize.view(x_flat_size)
            quantize = quantize.transpose(-1, 1).contiguous()

        cent_loss = (quantize.detach() - x).pow(2).mean()
        x_quant = x + (quantize - x).detach()

        return x_quant, cent_loss
