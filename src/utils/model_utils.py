import torch
import numpy as np
from src.metrics.inception_score import inception_score
from src.metrics.fid_score import calculate_fid_given_images


def count_parameters(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def toggle_grad(network, requires_grad):
    network.train(requires_grad)
    for p in network.parameters():
        if p.dtype == torch.FloatTensor:
            p.requires_grad_(requires_grad)


def zero_grad(network):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.zero_()


def get_grad_norm(network):
    grad_norm = 0.
    for p in network.parameters():
        if p.grad is not None:
            grad_norm += p.grad.pow(2).mean()
    return grad_norm


def clip_grad_ind_norm(network, max_norm=1., norm_type=torch._six.inf):
    for p in network.parameters():
        if p.grad is not None:
            if norm_type == torch._six.inf:
                norm = p.grad.data.abs().max()
            else:
                norm = p.grad.data.norm(norm_type)
            norm = torch.ones_like(p.grad.data) * max(norm, max_norm)
            p.grad.data.copy_(p.grad.data.max(norm))


def grad_noise(network, scale):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.copy_(p.grad + torch.randn_like(p.grad) * scale)


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
            lat_gen = generator(ztest)
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


def update_average(model_tgt, model_src, beta):
    param_dict_src = dict(model_src.named_parameters())

    for p_name, p_tgt in model_tgt.named_parameters():
        p_src = param_dict_src[p_name]
        assert(p_src is not p_tgt)
        p_tgt.copy_(beta*p_tgt + (1. - beta)*p_src)


def ca_seed(batch_size, n_filter, image_size, device, seed_value=1.0, all_channels=False):
    # The canvas
    seed = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    # The starting point of the wave
    if all_channels:
        seed[:, :, image_size // 2, image_size // 2] = seed_value
    else:
        seed[:, 0, image_size // 2, image_size // 2] = seed_value
    return seed


def checkerboard_seed(batch_size, n_filter, image_size, device):
    seed = np.zeros((batch_size, n_filter, image_size, image_size))
    cb = np.indices((1, 1, image_size, image_size)).sum(axis=0) % 2
    seed[:, 0, :, :] = np.tile(cb, (batch_size, 1, 1, 1))
    return torch.tensor(seed, device=device)


def grads_seed(batch_size, n_filter, image_size, device):
    seed = np.zeros((batch_size, n_filter, image_size, image_size))
    seed[:, :2, :, :] = np.indices((image_size, image_size))
    return torch.tensor(seed, device=device)


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
            setattr(self, k, np.asarray(v))

    def sample(self, n):
        idx = np.random.choice(self._size, n, False)
        batch = {k: getattr(self, k)[idx] for k in self._slot_names}
        batch = SamplePool(**batch, _parent=self, _parent_idx=idx)
        return batch

    def commit(self):
        for k in self._slot_names:
            getattr(self._parent, k)[self._parent_idx] = getattr(self, k)


# class MovingMean:
#     def __init__(self, mean_init=0.0, beta=0.9):
#         self.mean = mean_init
#         self.beta = beta
#
#     def update(self, obs):
#         self.mean = self.beta * self.mean + (1 - self.beta) * obs
#         return self.mean
#
#
# class Momentum:
#     def __init__(self, value_init=0.0, beta=0.9, alpha=0.1):
#         self.value = value_init
#         self.beta = beta
#         self.alpha = alpha * (1 - self.beta)
#         self.velocity = 0.
#
#     def update(self, grad):
#         self.velocity = self.beta * self.velocity + self.alpha * grad
#         self.value = self.value - self.velocity
#
#         return self.value
#
#
# class NesterovMomentum:
#     def __init__(self, value_init=0.0, beta=0.9, alpha=0.1):
#         self.value = value_init
#         self.beta = beta
#         self.alpha = alpha * (1 - self.beta)
#         self.velocity = 0.
#
#     def update(self, grad_func):
#         value_tmp = self.value - self.beta * self.velocity
#         self.velocity = self.beta * self.velocity + self.alpha * grad_func(value_tmp)
#         self.value = self.value - self.velocity
#
#         return self.value
#
#
# class RegEstimator:
#     def __init__(self, alpha=1e-2, beta1=0.5, beta2=0.9):
#         self.last_it = -1
#         self.alpha = alpha
#         self.beta1 = beta1
#         self.beta2 = beta2
#         self.m1 = 0.
#         self.m2 = 0.
#         self.total_it = 1
#
#     def update(self, curr_reg, target, obs):
#         grad = -2. * (target - obs)
#         self.m1 = self.beta1 * self.m1 + (1. - self.beta1) * grad
#         self.m2 = self.beta2 * self.m2 + (1. - self.beta2) * grad ** 2
#
#         m1_norm = self.m1 / (1. - self.beta1 ** self.total_it)
#         m2_norm = self.m2 / (1. - self.beta2 ** self.total_it)
#
#         next_reg = curr_reg - self.alpha * m1_norm / np.sqrt(m2_norm + 1e-8)
#
#         self.total_it += 1
#         return next_reg
#
#
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
