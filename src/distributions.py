import torch
from torch import distributions
import numpy as np
import torch.nn.functional as F


def get_zdist(dist_name, dim, device=None, mu=None, cov=None):
    # Get distribution
    if dist_name == 'uniform':
        low = -torch.ones(dim, device=device)
        high = torch.ones(dim, device=device)
        zdist = distributions.Uniform(low, high)
    elif dist_name == 'gauss':
        mu = torch.zeros(dim, device=device)
        scale = torch.ones(dim, device=device)
        zdist = distributions.Normal(mu, scale)
    elif dist_name == 'multigauss':
        mu = torch.zeros(dim, device=device) if mu is None else mu
        cov = torch.eye(dim, device=device) if cov is None else cov
        zdist = distributions.MultivariateNormal(mu, cov)
    else:
        raise NotImplementedError

    # Add dim attribute
    zdist.dim = dim

    return zdist


def get_ydist(n_labels, device=None):
    logits = torch.zeros(n_labels, device=device)
    ydist = distributions.categorical.Categorical(logits=logits)

    # Add n_labels attribute
    ydist.n_labels = n_labels

    return ydist


def interpolate_sphere(z1, z2, t):
    p = (z1 * z2).sum(dim=-1, keepdim=True)
    p = p / z1.pow(2).sum(dim=-1, keepdim=True).sqrt()
    p = p / z2.pow(2).sum(dim=-1, keepdim=True).sqrt()
    omega = torch.acos(p)
    s1 = torch.sin((1-t)*omega)/torch.sin(omega)
    s2 = torch.sin(t*omega)/torch.sin(omega)
    z = s1 * z1 + s2 * z2

    return z


def cov(m, rowvar=True, inplace=False):
    '''Estimate a covariance matrix given data.

    Covariance indicates the level to which two variables vary together.
    If we examine N-dimensional samples, `X = [x_1, x_2, ... x_N]^T`,
    then the covariance matrix element `C_{ij}` is the covariance of
    `x_i` and `x_j`. The element `C_{ii}` is the variance of `x_i`.

    Args:
        m: A 1-D or 2-D array containing multiple variables and observations.
            Each row of `m` represents a variable, and each column a single
            observation of all those variables.
        rowvar: If `rowvar` is True, then each row represents a
            variable, with observations in the columns. Otherwise, the
            relationship is transposed: each column represents a variable,
            while the rows contain observations.

    Returns:
        The covariance matrix of the variables.
    '''
    if m.dim() > 2:
        raise ValueError('m has more than 2 dimensions')
    if m.dim() < 2:
        m = m.view(1, -1)
    if not rowvar and m.size(0) != 1:
        m = m.t()
    # m = m.type(torch.double)  # uncomment this line if desired
    fact = 1.0 / (m.size(1) - 1)
    if inplace:
        m -= torch.mean(m, dim=1, keepdim=True)
    else:
        m = m - torch.mean(m, dim=1, keepdim=True)
    mt = m.t()  # if complex: mt = m.t().conj()
    return fact * m.matmul(mt).squeeze()


def get_multigauss_params(samples):
    zdist_mu = samples.mean(dim=0)
    zdist_cov = cov(samples, False, False)
    return zdist_mu, zdist_cov


def update_multigauss_params(dim, zdist, zdist_mu, zdist_cov, samples, momentum):
    with torch.no_grad():
        del zdist
        zdist_mu_new, zdist_cov_new = get_multigauss_params(samples)
        zdist_mu = momentum * zdist_mu + (1-momentum) * zdist_mu_new
        zdist_cov = momentum * zdist_cov + (1-momentum) * zdist_cov_new
        zdist = get_zdist('multigauss', dim, samples.device, zdist_mu, zdist_cov)
        return zdist, zdist_mu, zdist_cov


def save_multigauss_params(zdist_mu, zdist_cov, save_path):
    zdist_mu = zdist_mu.data.cpu().numpy()
    zdist_cov = zdist_cov.data.cpu().numpy()
    np.save(save_path + '/multigauss_params_mu.npy', zdist_mu)
    np.save(save_path + '/multigauss_params_cov.npy', zdist_cov)


def load_multigauss_params(save_path, dim, device):
    try:
        zdist_mu = torch.tensor(np.load(save_path + '/multigauss_params_mu.npy'), device=device)
        zdist_cov = torch.tensor(np.load(save_path + '/multigauss_params_cov.npy'), device=device)
    except IOError:
        zdist_mu = torch.zeros(dim, device=device)
        zdist_cov = 1e-3 * torch.eye(dim, device=device)
    return zdist_mu, zdist_cov
