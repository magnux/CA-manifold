import numpy as np
import torch
from torch.nn import functional as F
from src.layers.posencoding import cos_pos_encoding_nd

CE_ZERO = np.log(0.5)


def compute_gan_loss(d_out, target, gan_type='new_relu'):

    if gan_type == 'standard':
        target = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.binary_cross_entropy_with_logits(d_out, target)
    elif gan_type == 'wgan':
        loss = ((1 - 2*target) * d_out).mean()
    elif gan_type == 'abs':
        loss = ((2*target - 1) * d_out).mean()
        loss = loss.abs() if target == 1 else loss
    elif gan_type == 'mse':
        target = d_out.new_full(size=d_out.size(), fill_value=target)
        loss = F.mse_loss(d_out.clamp(0., 1.),  target)
    elif gan_type == 'rectified':
        loss = (1 - 2*target) * (d_out.clamp(-1, 1) + 1e-3 * d_out).mean()
    elif gan_type == 'tanh':
        loss = (1 - 2*target) * (0.1 * d_out).tanh().mean()
    elif gan_type == 'dssqrt':
        loss = (1 - 2*target) * (d_out * (d_out.abs() + 1e-8).rsqrt()).mean()
    elif gan_type == 'saddle':
        loss = (1 - 2*target) * (d_out.pow(3) / d_out.abs().pow(1.5)).mean()
    elif gan_type == 'relu':
        loss = F.relu((1 - 2*target) * d_out).mean()
    elif gan_type == 'softplus':
        loss = F.softplus((1 - 2*target) * d_out).mean()
    elif gan_type == 'double_softplus':
        loss = (F.softplus((1 - 2*target) * d_out) - F.softplus((2*target - 1) * d_out)).mean()
    elif gan_type == 'new_mse':
        target = d_out.new_full(size=d_out.size(), fill_value=(1 - 2*target))
        loss = F.mse_loss(d_out, target)
    elif gan_type == 'new_relu':
        loss = F.relu(1 + ((1 - 2*target) * d_out)).mean()
    elif gan_type == 'interstella':
        x = (1 - 2*target) * d_out
        loss = torch.maximum((torch.maximum(x, torch.ones_like(x) * -0.5554) + 0.5555).log() + 1, torch.sigmoid(x * 5)).mean()
    else:
        raise NotImplementedError

    return loss


def compute_grad_reg(d_out, d_in, norm_type=2, margin=0, mask=None):
    batch_size = d_in[0].size(0) if isinstance(d_in, list) else d_in.size(0)
    grad_d_in = torch.autograd.grad(outputs=d_out.sum(), inputs=d_in,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    # assert(grad_d_in.size() == d_in.size())

    if norm_type == 1:
        grad_d_in = grad_d_in.abs()
    elif norm_type == 2:
        grad_d_in = grad_d_in.pow(2)
    elif norm_type == 4:
        grad_d_in = grad_d_in.pow(4)
    elif norm_type == 'sqrt':
        grad_d_in = (grad_d_in.abs() + 1e-8).sqrt()
    elif norm_type == 'log':
        grad_d_in = (grad_d_in.abs() + 1.).log()
    elif norm_type == 'exp':
        grad_d_in = grad_d_in.exp() - 1.
    elif norm_type == 'neg':
        grad_d_in = -1 * grad_d_in.abs()

    if norm_type == 'cov':
        # grad_d_in_norm = grad_d_in.pow(2).sum(1).mean()
        # grad_d_in_cov = grad_d_in.reshape(batch_size, -1).var(dim=1).sqrt()
        # grad_d_in_cov = grad_d_in_cov.view([1, batch_size]) @ grad_d_in_cov.view([batch_size, 1])
        grad_d_in_cov = grad_d_in.reshape(batch_size, -1)
        grad_d_in_cov = grad_d_in_cov @ grad_d_in_cov.t()
        grad_d_in_cov = grad_d_in_cov * (1 - torch.eye(batch_size, device=grad_d_in_cov.device))
        grad_d_in_cov = grad_d_in_cov.mean()
        return grad_d_in_cov
    else:
        if mask is not None:
            grad_d_in = grad_d_in * mask
        reg = grad_d_in.view(batch_size, -1)
        if norm_type == 'sqrt':
            reg = reg / reg.size(1)
        if margin > 0:
            reg = torch.relu(reg - margin)
        reg = reg.sum(1).mean()

        return reg


def compute_dir_grad_reg(d_out, d_out_inv, d_in, d_in_inv=None, margin=0):
    batch_size = d_in[0].size(0) if isinstance(d_in, list) else d_in.size(0)
    if d_in_inv is None:
        d_in_inv = d_in
    grad_d_in_l = torch.autograd.grad(outputs=d_out.sum(), inputs=d_in,
                                    create_graph=True, retain_graph=True, only_inputs=True)

    grad_d_in_inv_l = torch.autograd.grad(outputs=d_out_inv.sum(), inputs=d_in_inv,
                                        create_graph=True, retain_graph=True, only_inputs=True)
    reg = 0.
    for grad_d_in, grad_d_in_inv in zip(grad_d_in_l, grad_d_in_inv_l):
        grad_d_in = grad_d_in.reshape(batch_size, -1)
        # grad_d_in = torch.fft.fft2(grad_d_in, norm='ortho')
        # grad_d_in = torch.cat([grad_d_in.real.to(torch.float32), grad_d_in.imag.to(torch.float32)], dim=1)
        grad_d_in_inv = grad_d_in_inv.reshape(batch_size, -1) #.detach()
        # grad_d_in_inv = torch.fft.fft2(grad_d_in_inv, norm='ortho')
        # grad_d_in_inv = torch.cat([grad_d_in_inv.real.to(torch.float32), grad_d_in_inv.imag.to(torch.float32)], dim=1)
        grad_d_in_diff = (grad_d_in - grad_d_in_inv).pow(2)
        if margin > 0:
            grad_d_in_diff = torch.relu(grad_d_in_diff - margin)
        reg += grad_d_in_diff.mean()

    return reg


def compute_double_grad_reg(d_out, d_in):
    grad_d_in = torch.autograd.grad(outputs=d_out.sum(), inputs=d_in,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    reg = grad_d_in.pow(2).sum()
    double_grad_d_in = torch.autograd.grad(outputs=reg, inputs=d_in,
                                           create_graph=True, retain_graph=True, only_inputs=True)[0]

    reg = reg + double_grad_d_in.pow(2).sum()

    return reg


def compute_hinted_sample(g_in, d_out, target=1, gan_type='softplus'):
    d_loss = compute_gan_loss(d_out, target, gan_type)
    grad_g_in = torch.autograd.grad(outputs=d_loss, inputs=g_in,
                                    create_graph=True, retain_graph=True, only_inputs=True)[0]
    assert (grad_g_in.size() == g_in.size())

    hinted_sample = g_in - grad_g_in

    return hinted_sample.detach()


def update_reg_params(reg_every, reg_every_target, reg_param, reg_param_target, reg_loss, reg_loss_target,
                      loss_dis=None, update_every=True, maximize=True, lr=1e-2):

    if loss_dis is not None:
        # Emergency break, in case the discriminator had slowly slip through the fence
        if loss_dis < 0.1:
            if update_every:
                reg_every = 1
            return reg_every, reg_param

    # reg_param update
    delta_reg = reg_loss_target - reg_loss
    reg_scale = 2 * reg_param / (reg_loss_target + reg_loss + 1e-32)
    reg_update = lr * reg_scale * delta_reg
    if maximize:
        reg_param += reg_update
        reg_param = np.clip(reg_param, 0.5 * reg_loss_target, reg_param_target)
    else:
        reg_param -= reg_update
        reg_param = np.clip(reg_param, reg_param_target, 2 * reg_loss_target)

    # reg_every update
    if update_every:
        reg_ratio = (reg_loss / (reg_loss_target + 1e-32))
        if reg_ratio <= 1.:
            reg_every += 0.5
        elif reg_ratio > 2.:
            reg_every /= 2

        reg_every = np.clip(reg_every, 1, reg_every_target)

    return reg_every, reg_param


def compute_pl_reg(g_out, g_in, pl_mean, beta=0.99, alt_pl=None, reg_factor=1., noise_mode='normal', out_mode='mul'):
    if g_out.dim() == 2:
        g_out = g_out.unsqueeze(1)

    if noise_mode == 'normal':
        pl_noise = torch.randn_like(g_out)
    elif noise_mode == 'dct':
        dc_enc = cos_pos_encoding_nd(int(g_out.size(2)), int(g_out.dim()) - 2)
        dc_enc = torch.cat([dc_enc] * g_out.size(0), dim=0).to(g_out.device)
        pl_noise = torch.randn([dc_enc.size(0), dc_enc.size(1)] + [1 for _ in range(2, g_out.dim())], device=g_out.device)
        pl_noise = (pl_noise * dc_enc).mean(dim=1, keepdim=True)
    space_sqrt = np.sqrt(np.prod([g_out.size(i) for i in range(2, g_out.dim())]))
    if out_mode == 'mul':
        pl_noise = pl_noise / space_sqrt
        outputs = (g_out * pl_noise).sum()
    elif out_mode == 'add':
        pl_noise = pl_noise / (10 * space_sqrt)
        g_out = g_out / space_sqrt
        outputs = (g_out + pl_noise).sum()

    pl_grads = torch.autograd.grad(outputs=outputs, inputs=g_in,
                                   create_graph=True, retain_graph=True, only_inputs=True)[0]

    pl_lengths = ((pl_grads ** 2).mean(dim=1) + 1e-8).sqrt()
    if alt_pl is None:
        pl_reg = ((pl_lengths - pl_mean) ** 2).mean()
    else:
        pl_reg = ((pl_lengths - alt_pl) ** 2).mean()
    pl_reg *= reg_factor

    avg_pl_length = np.mean(pl_lengths.detach().cpu().numpy())
    new_pl_mean = pl_mean * beta + (1 - beta) * avg_pl_length

    return pl_reg, new_pl_mean


def update_ada_augment_p(current_p, logits_sign_mean, batch_size, ada_target=0.2, kimgs=5e5):
    adjust = np.sign(logits_sign_mean - ada_target) * batch_size / kimgs
    return F.relu(current_p + adjust)


def update_g_factor(g_factor, lab_dis_sign, sign_mean_target, maximize=False, lr=1e-2):
    delta_reg = sign_mean_target - lab_dis_sign
    reg_update = lr * delta_reg

    if maximize:
        if reg_update < 0.:
            reg_update *= 2
        g_factor += reg_update
    else:
        if reg_update > 0.:
            reg_update *= 2
        g_factor -= reg_update

    g_factor = np.clip(g_factor, 1e-16, 1.)
    return g_factor


def update_g_factors(g_factor_enc, g_factor_dec, labs_dis_enc_sign, labs_dis_dec_sign, sign_mean_target, maximize=False, lr=1e-2):
    g_factor_enc = update_g_factor(g_factor_enc, labs_dis_enc_sign, sign_mean_target, maximize, lr)
    g_factor_dec = update_g_factor(g_factor_dec, labs_dis_dec_sign, sign_mean_target, maximize, lr)
    return g_factor_enc, g_factor_dec


def gram_matrix_1d(x):
    b, f, w = x.size()
    G = torch.bmm(x, x.permute(0, 2, 1))
    return G.div(f * w)


def gram_matrix_2d(x):
    b, f, h, w = x.size()
    x = x.view(b, f, h * w)
    G = torch.bmm(x, x.permute(0, 2, 1))
    return G.div(f * h * w)


def cross_entropy_distance(out, target, mode='L1'):
    if mode == 'L1':
        dist = (target - out).abs()
    elif mode == 'L2':
        dist = (target - out).pow(2)
    elif mode == 'CE':
        target_idx = torch.argmax(target, dim=1)
        dist = F.cross_entropy(out, target_idx, reduction='none')
    elif mode == 'BCE':
        dist = 10 * F.binary_cross_entropy(out, target.detach(), reduction='none')
    else:
        raise NotImplementedError
    loss = ((1. + (dist + 1e-4).exp()).log() + CE_ZERO)
    return loss.mean()


def log_sum_exp(x):
    """ numerically stable log_sum_exp implementation that prevents overflow """
    # TF ordering
    axis  = len(x.size()) - 1
    m, _  = torch.max(x, dim=axis)
    m2, _ = torch.max(x, dim=axis, keepdim=True)
    return m + torch.log(torch.sum(torch.exp(x - m2), dim=axis))


def log_prob_from_logits(x):
    """ numerically stable log_softmax implementation that prevents overflow """
    # TF ordering
    axis = len(x.size()) - 1
    m, _ = torch.max(x, dim=axis, keepdim=True)
    return x - m - torch.log(torch.sum(torch.exp(x - m), dim=axis, keepdim=True))


def discretized_mix_logistic_loss(output, target):
    """ log-likelihood for mixture of discretized logistics, assumes the data has been rescaled to [-1,1] interval """
    # Pytorch ordering
    target = target.permute(0, 2, 3, 1)
    output = output.permute(0, 2, 3, 1)
    xs = [int(y) for y in target.size()]
    ls = [int(y) for y in output.size()]

    # here and below: unpacking the params of the mixture of logistics
    nr_mix = int(ls[-1] / 10)
    logit_probs = output[:, :, :, :nr_mix]
    output = output[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])  # 3 for mean, scale, coef
    means = output[:, :, :, :, :nr_mix]
    # log_scales = torch.max(l[:, :, :, :, nr_mix:2 * nr_mix], -7.)
    log_scales = torch.clamp(output[:, :, :, :, nr_mix:2 * nr_mix], min=-7.)

    coeffs = torch.tanh(output[:, :, :, :, 2 * nr_mix:3 * nr_mix])
    # here and below: getting the means and adjusting them based on preceding
    # sub-pixels
    target = target.contiguous()
    target = target.unsqueeze(-1) + torch.zeros(xs + [nr_mix], device=target.device, requires_grad=False)
    m2 = (means[:, :, :, 1, :] + coeffs[:, :, :, 0, :]
          * target[:, :, :, 0, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    m3 = (means[:, :, :, 2, :] + coeffs[:, :, :, 1, :] * target[:, :, :, 0, :] +
          coeffs[:, :, :, 2, :] * target[:, :, :, 1, :]).view(xs[0], xs[1], xs[2], 1, nr_mix)

    means = torch.cat((means[:, :, :, 0, :].unsqueeze(3), m2, m3), dim=3)
    centered_x = target - means
    inv_stdv = torch.exp(-log_scales)
    plus_in = inv_stdv * (centered_x + 1. / 255.)
    cdf_plus = torch.sigmoid(plus_in)
    min_in = inv_stdv * (centered_x - 1. / 255.)
    cdf_min = torch.sigmoid(min_in)
    # log probability for edge case of 0 (before scaling)
    log_cdf_plus = plus_in - F.softplus(plus_in)
    # log probability for edge case of 255 (before scaling)
    log_one_minus_cdf_min = -F.softplus(min_in)
    cdf_delta = cdf_plus - cdf_min  # probability for all other cases
    mid_in = inv_stdv * centered_x
    # log probability in the center of the bin, to be used in extreme cases
    # (not actually used in our code)
    log_pdf_mid = mid_in - log_scales - 2. * F.softplus(mid_in)

    # now select the right output: left edge case, right edge case, normal
    # case, extremely low prob case (doesn't actually happen for us)

    # this is what we are really doing, but using the robust version below for extreme cases in other applications and to avoid NaN issue with tf.select()
    # log_probs = tf.select(x < -0.999, log_cdf_plus, tf.select(x > 0.999, log_one_minus_cdf_min, tf.log(cdf_delta)))

    # robust version, that still works if probabilities are below 1e-5 (which never happens in our code)
    # tensorflow backpropagates through tf.select() by multiplying with zero instead of selecting: this requires use to use some ugly tricks to avoid potential NaNs
    # the 1e-12 in tf.maximum(cdf_delta, 1e-12) is never actually used as output, it's purely there to get around the tf.select() gradient issue
    # if the probability on a sub-pixel is below 1e-5, we use an approximation
    # based on the assumption that the log-density is constant in the bin of
    # the observed sub-pixel value

    inner_inner_cond = (cdf_delta > 1e-5).float()
    inner_inner_out = inner_inner_cond * torch.log(torch.clamp(cdf_delta, min=1e-12)) + (1. - inner_inner_cond) * (log_pdf_mid - np.log(127.5))
    inner_cond = (target > 0.999).float()
    inner_out = inner_cond * log_one_minus_cdf_min + (1. - inner_cond) * inner_inner_out
    cond = (target < -0.999).float()
    log_probs = cond * log_cdf_plus + (1. - cond) * inner_out
    log_probs = torch.sum(log_probs, dim=3) + log_prob_from_logits(logit_probs)

    # return -torch.sum(log_sum_exp(log_probs))
    return -torch.mean(log_sum_exp(log_probs))


def to_one_hot(tensor, n, fill_with=1.):
    # we perform one hot encode with respect to the last axis
    one_hot = torch.zeros(tensor.size() + (n,), device=tensor.device)
    one_hot.scatter_(len(tensor.size()), tensor.unsqueeze(-1), fill_with)
    return one_hot


def sample_from_discretized_mix_logistic(output, nr_mix):
    # Pytorch ordering
    output = output.permute(0, 2, 3, 1)
    ls = [int(y) for y in output.size()]
    xs = ls[:-1] + [3]

    # unpack parameters
    logit_probs = output[:, :, :, :nr_mix]
    output = output[:, :, :, nr_mix:].contiguous().view(xs + [nr_mix * 3])
    # sample mixture indicator from softmax
    temp = torch.empty(logit_probs.size(), device=output.device)
    temp.uniform_(1e-5, 1. - 1e-5)
    temp = logit_probs.data - torch.log(- torch.log(temp))
    _, argmax = temp.max(dim=3)

    one_hot = to_one_hot(argmax, nr_mix)
    sel = one_hot.view(xs[:-1] + [1, nr_mix])
    # select logistic parameters
    means = torch.sum(output[:, :, :, :, :nr_mix] * sel, dim=4)
    log_scales = torch.clamp(torch.sum(
        output[:, :, :, :, nr_mix:2 * nr_mix] * sel, dim=4), min=-7.)
    coeffs = torch.sum(torch.tanh(
        output[:, :, :, :, 2 * nr_mix:3 * nr_mix]) * sel, dim=4)
    # sample from logistic & clip to interval
    # we don't actually round to the nearest 8bit value when sampling
    u = torch.empty(means.size(), device=output.device)
    u.uniform_(1e-5, 1. - 1e-5)
    x = means + torch.exp(log_scales) * (torch.log(u) - torch.log(1. - u))
    x0 = torch.clamp(torch.clamp(x[:, :, :, 0], min=-1.), max=1.)
    x1 = torch.clamp(torch.clamp(
        x[:, :, :, 1] + coeffs[:, :, :, 0] * x0, min=-1.), max=1.)
    x2 = torch.clamp(torch.clamp(
        x[:, :, :, 2] + coeffs[:, :, :, 1] * x0 + coeffs[:, :, :, 2] * x1, min=-1.), max=1.)

    out = torch.cat([x0.view(xs[:-1] + [1]), x1.view(xs[:-1] + [1]), x2.view(xs[:-1] + [1])], dim=3)
    # put back in Pytorch ordering
    out = out.permute(0, 3, 1, 2).contiguous()
    return out


def vae_gaussian_kl_loss(mu, log_var):
    kl_div = - 0.5 * torch.sum(1 + log_var - mu.pow(2) - log_var.exp(), dim=1)
    return kl_div.mean()


def vae_sample_gaussian(mu, log_var):
    std = torch.exp(log_var / 2)
    eps = torch.randn_like(std)
    return mu + eps * std


def age_gaussian_kl_loss(samples, kl_dir_pq=False):
    samples_var = samples.var(0)
    samples_mean = samples.mean(0)

    if kl_dir_pq:
        # mu_1 = 0; sigma_1 = 1
        t1 = (1 + samples_mean.pow(2)) / (2 * samples_var.pow(2))
        t2 = samples_var.log()
    else:
        # mu_2 = 0; sigma_2 = 1
        t1 = (samples_var.pow(2) + samples_mean.pow(2)) / 2
        t2 = -samples_var.log()

    KL = (t1 + t2 - 0.5).mean()

    return KL


def rage_gaussian_loss(x, target=1):
    if target == 1:
        return (1 - (-(x ** 2)).exp() + 1e-4 * x.abs()).mean()
    else:
        return ((-(x ** 2)).exp() - 1e-4 * x.abs()).mean()


def fft_mse_loss(out, target, margin=0.):
    out_fft = torch.fft.fft2(out, norm='ortho')
    out_fft = torch.cat([out_fft.real.to(torch.float32), out_fft.imag.to(torch.float32)], dim=1)

    target_fft = torch.fft.fft2(target, norm='ortho')
    target_fft = torch.cat([target_fft.real.to(torch.float32), target_fft.imag.to(torch.float32)], dim=1)

    if margin > 0.:
        return F.relu((target_fft - out_fft).pow(2) - margin).mean()
    else:
        return F.mse_loss(out_fft, target_fft)
