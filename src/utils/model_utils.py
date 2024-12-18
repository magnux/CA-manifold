import torch
import torch.nn.functional as F
import numpy as np
from src.metrics.inception_score import inception_score
from src.metrics.fid_score import calculate_fid_given_images
from functools import partial
from collections import OrderedDict
from src.layers.gaussiansmoothing import get_gaussian_kernel

GAUSSIAN_KERNEL = None


def count_parameters(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def toggle_grad(network, requires_grad):
    network.train(requires_grad)
    for p in network.parameters():
        if p.dtype == torch.float32:
            p.requires_grad_(requires_grad)


def zero_grad(network):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.zero_()


def get_grad_norm(network):
    grad_norm = 0.
    for p in network.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm(2)
    return grad_norm


def get_mean_grad_norm(network):
    grad_norm = 0.
    n_grads = 0
    for p in network.parameters():
        if p.grad is not None:
            grad_norm += p.grad.norm(2)
            n_grads += 1
    return grad_norm / n_grads


def get_grads_stats(network):
    stats_dict = OrderedDict()
    for name, p in network.named_parameters():
        if p.grad is not None:
            stats_dict[name] = get_tensor_stats(p.grad)
    return stats_dict


def get_tensor_stats(tensor):
    return tensor.norm(2).item(), tensor.abs().max().item(), tensor.std().item()


def clip_grad_ind_norm(network, max_norm=1., norm_type=torch._six.inf):
    for p in network.parameters():
        if p.grad is not None:
            if norm_type == torch._six.inf:
                norm = p.grad.data.abs().max()
            else:
                norm = p.grad.data.norm(norm_type)
            norm = torch.ones_like(p.grad.data) * max(norm, max_norm)
            p.grad.data.copy_(p.grad.data.max(norm))


def grad_dither(network, g_factor=0.1):
    mean_p_grad = get_mean_grad_norm(network)
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.copy_(p.grad + g_factor * (mean_p_grad - p.grad.norm(2)).clamp(0, 1) * torch.rand_like(p.grad) * torch.randn_like(p.grad))


def grad_dither_hook(g_factor):
    def _grad_dither_hook(grad, g_factor=1.):
        with torch.no_grad():
            return grad + g_factor * torch.rand_like(grad) * torch.rand_like(grad)
    return partial(_grad_dither_hook, g_factor=g_factor)


def grad_mult(network, g_factor):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.copy_(g_factor * p.grad)


def grad_mult_hook(g_factor):
    def _grad_mult_hook(grad, g_factor=1):
        with torch.no_grad():
            return g_factor * grad
    return partial(_grad_mult_hook, g_factor=g_factor)


def image_grad_smooth_hook(channels, g_factor=0.5):
    global GAUSSIAN_KERNEL

    if GAUSSIAN_KERNEL is None:
        GAUSSIAN_KERNEL = get_gaussian_kernel(channels, 3, 0.5)

    def _grad_smooth_hook(grad, channels=channels, g_factor=1):
        global GAUSSIAN_KERNEL

        GAUSSIAN_KERNEL = GAUSSIAN_KERNEL.to(grad.device)
        with torch.no_grad():
            return g_factor * grad + (1. - g_factor) * F.conv2d(grad, GAUSSIAN_KERNEL, padding=1, groups=channels)
    return partial(_grad_smooth_hook, channels=channels, g_factor=g_factor)


def grad_ema_update(network, momentum=0.9, mix_ratio=0.5):
    for p in network.parameters():
        if p.grad is not None:
            if not hasattr(p, 'grad_ema'):
                p.grad_ema = p.grad.clone()
            else:
                p.grad_ema = momentum * p.grad_ema + (1 - momentum) * p.grad.clone()
                p.grad = mix_ratio * p.grad + (1 - mix_ratio) *p.grad_ema.clone()


def bkp_grad(network, bkp_name):
    for p in network.parameters():
        if p.grad is not None:
            if not hasattr(p, 'grad_bkp'):
                p.grad_bkp = {}
            p.grad_bkp[bkp_name] = p.grad


def copy_grad_bkp(network, bkp_name):
    for p in network.parameters():
        if p.grad is not None:
            if hasattr(p, 'grad_bkp'):
                p.grad = p.grad_bkp[bkp_name]
            else:
                print('warning, no grad_bkp found!')


def del_grad_bkp(network):
    for p in network.parameters():
        if p.grad is not None:
            if hasattr(p, 'grad_bkp'):
                for k in list(p.grad_bkp.keys()):
                    del p.grad_bkp[k]
                delattr(p, 'grad_bkp')
            else:
                print('warning, no grad_bkp found!')


def apply_grad_bkp(network, *args):
    for p in network.parameters():
        if p.grad is not None:
            if hasattr(p, 'grad_bkp'):
                if len(args) == 2:
                    bkp_name = args[0]
                    func = args[1]
                    p.grad_bkp[bkp_name] = func(p.grad_bkp[bkp_name])
                elif len(args) == 3:
                    bkp_name_a = args[0]
                    bkp_name_b = args[1]
                    func = args[2]
                    p.grad_bkp[bkp_name_a] = func(p.grad_bkp[bkp_name_a], p.grad_bkp[bkp_name_b])
                    p.grad_bkp[bkp_name_b] = p.grad_bkp[bkp_name_a]
                else:
                    raise RuntimeError('args not suported: %s' % str(args))
            else:
                print('warning, no grad_bkp found!')


def make_safe(tens, min, max):
    tens = torch.where(torch.isnan(tens), torch.zeros_like(tens), tens)
    tens = tens.clamp(min=min, max=max)
    return tens


def make_grad_safe(params, min, max):
    for p in params:
        if p.grad is not None:
            p.grad.data.copy_(make_safe(p.grad, min, max))


def compute_inception_score(generator, decoder, inception_sample_size, fid_sample_size, batch_size, zdist, ydist, fid_real_samples, device, npass=1):
    imgs = []
    while len(imgs) < inception_sample_size:
        ztest = zdist.sample((batch_size,))
        if ydist is not None:
            ytest = ydist.sample((batch_size,))
            lat_gen = generator(ztest, ytest)
        else:
            if generator is not None:
                lat_gen = generator(ztest)
            else:
                lat_gen = ztest
        out_embs = [None]
        samples = None
        for _ in npass:
            samples, out_embs, _ = decoder(lat_gen, out_embs[-1])
        samples = [s.data.cpu().numpy() for s in samples]
        imgs.extend(samples)

    inception_imgs = imgs[:inception_sample_size]
    score, score_std = inception_score(
        inception_imgs, device=device, resize=True, splits=10, batch_size=batch_size
    )

    fid_imgs = np.array(imgs[:fid_sample_size])
    fid = calculate_fid_given_images(
        fid_real_samples, fid_imgs, batch_size=batch_size, cuda='cuda' in device.type
    )

    return score, score_std, fid


def update_network_average(network_tgt, network_src, beta):
    with torch.no_grad():
        param_dict_src = dict(network_src.named_parameters())

        for p_name, p_tgt in network_tgt.named_parameters():
            p_src = param_dict_src[p_name]
            assert(p_src is not p_tgt)
            p_tgt.copy_(beta * p_tgt + (1. - beta) * p_src)


def ca_seed(batch_size, n_filter, image_size, device, seed_value=1.0, all_channels=False):
    # The canvas
    seed = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    # The starting point of the wave
    if all_channels:
        seed[:, :, image_size // 2, image_size // 2] = seed_value
    else:
        seed[:, 0, image_size // 2, image_size // 2] = seed_value
    return seed


def ca_seed3d(batch_size, n_filter, image_size, device, seed_value=1.0, all_channels=False):
    # The canvas
    seed = torch.zeros(batch_size, n_filter, image_size, image_size, image_size, device=device)
    # The starting point of the wave
    if all_channels:
        seed[:, :, image_size // 2, image_size // 2, image_size // 2] = seed_value
    else:
        seed[:, 0, image_size // 2, image_size // 2, image_size // 2] = seed_value
    return seed


def checkerboard_seed(batch_size, n_filter, image_size, device):
    seed = np.zeros((batch_size, n_filter, image_size, image_size))
    cb = np.indices((1, 1, image_size, image_size)).sum(axis=0) % 2
    seed[:, 0:1, :, :] = np.tile(cb, (batch_size, 1, 1, 1))
    return torch.tensor(seed, device=device, dtype=torch.float32)


def grads_seed(batch_size, n_filter, image_size, device):
    seed = np.zeros((batch_size, n_filter, image_size, image_size))
    seed[:, :2, :, :] = np.indices((image_size, image_size))
    return torch.tensor(seed, device=device)


def normalize_2nd_moment(x, dim=1, eps=1e-8):
    return x * (x.square().mean(dim=dim, keepdim=True) + eps).rsqrt()


class SamplePool:
    def __init__(self, *, _parent=None, _parent_idx=None, **slots):
        self._parent = _parent
        self._parent_idx = _parent_idx
        self._slot_names = slots.keys()
        self._size = None
        for k, v in slots.items():
            if self._size is None:
                self._size = len(v)
            assert self._size == len(v)
            # setattr(self, k, np.asarray(v))
            setattr(self, k, v)

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


class MovingMean:
    def __init__(self, mean_init=0.0, beta=0.9):
        self.mean = mean_init
        self.beta = beta

    def update(self, obs):
        self.mean = self.beta * self.mean + (1 - self.beta) * obs
        return self.mean


class Momentum:
    def __init__(self, value_init=0.0, beta=0.9, alpha=0.1):
        self.value = value_init
        self.beta = beta
        self.alpha = alpha * (1 - self.beta)
        self.velocity = 0.

    def update(self, grad):
        self.velocity = self.beta * self.velocity + self.alpha * grad
        self.value = self.value - self.velocity

        return self.value


class NesterovMomentum:
    def __init__(self, value_init=0.0, beta=0.9, alpha=0.1):
        self.value = value_init
        self.beta = beta
        self.alpha = alpha * (1 - self.beta)
        self.velocity = 0.

    def update(self, grad_func):
        value_tmp = self.value - self.beta * self.velocity
        self.velocity = self.beta * self.velocity + self.alpha * grad_func(value_tmp)
        self.value = self.value - self.velocity

        return self.value


# class KalmanFilter:
#     def __init__(self, total_it, Q_init=1e-2, R_init=1e-2, xhat_init=0.0, P_init=1.0):
#         self.total_it = total_it
#         self.Q = Q_init  # process variance
#         self.R = R_init  # estimate of measurement variance
#
#         # allocate space for arrays
#         self.xhat = np.zeros(total_it)  # a posteriori estimate of x
#         self.P = np.zeros(total_it)  # a posteriori error estimate
#         self.xhatminus = np.zeros(total_it)  # a priori estimate of x
#         self.Pminus = np.zeros(total_it)  # a priori error estimate
#         self.K = np.zeros(total_it)  # gain or blending factor
#
#         self.xhat[0] = xhat_init
#         self.P[0] = P_init
#         self.last_it = 0
#
#     def update_kf(self, it, obs):
#         if it - self.last_it > 1:
#             self.xhat[self.last_it + 1: it] = self.xhat[self.last_it]
#             self.P[self.last_it + 1: it] = self.P[self.last_it]
#             self.xhatminus[self.last_it + 1: it] = self.xhatminus[self.last_it]
#             self.Pminus[self.last_it + 1: it] = self.Pminus[self.last_it]
#             self.K[self.last_it + 1: it] = self.K[self.last_it]
#
#         # time update
#         self.xhatminus[it] = self.xhat[it - 1]
#         self.Pminus[it] = self.P[it - 1] + self.Q
#
#         # measurement update
#         self.K[it] = self.Pminus[it] / (self.Pminus[it] + self.R)
#         self.xhat[it] = self.xhatminus[it] + self.K[it] * (obs - self.xhatminus[it])
#         self.P[it] = (1 - self.K[it]) * self.Pminus[it]
#
#         self.last_it = it
#         return self.xhat[it]
