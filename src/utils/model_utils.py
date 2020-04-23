import torch
import numpy as np
from src.metrics.inception_score import inception_score
from src.metrics.fid_score import calculate_fid_given_images
from src.utils.loss_utils import make_safe


def count_parameters(network):
    return sum(p.numel() for p in network.parameters() if p.requires_grad)


def toggle_grad(model, requires_grad):
    model.train(requires_grad)
    for p in model.parameters():
        if p.dtype == torch.FloatTensor:
            p.requires_grad_(requires_grad)


def zero_grad(network):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.zero_()


def make_grads_safe(network):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.copy_(make_safe(p.grad))


def get_nsamples(data_loader, N, device):
    x = []
    y = []
    n = 0
    while n < N:
        x_next, y_next = next(iter(data_loader))
        x.append(x_next)
        y.append(y_next)
        n += x_next.size(0)
    x = torch.cat(x, dim=0)[:N]
    y = torch.cat(y, dim=0)[:N]

    x = x.to(device)
    y = y.to(device)

    return x, y


def compute_inception_score(generator, decoder, inception_sample_size, fid_sample_size, batch_size, zdist, ydist, fid_real_samples, device):
    imgs = []
    while len(imgs) < inception_sample_size:
        ztest = zdist.sample((batch_size,))
        ytest = ydist.sample((batch_size,))

        qlat_gen = generator(ztest, ytest)
        samples, _, _ = decoder(qlat_gen)
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
