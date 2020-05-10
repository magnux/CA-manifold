import torch
import numpy as np
from src.metrics.inception_score import inception_score
from src.metrics.fid_score import calculate_fid_given_images


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


def make_safe(tens, min=-1e3, max=1e3):
    tens = torch.where(torch.isnan(tens), torch.zeros_like(tens), tens)
    tens = tens.clamp(min=min, max=max)
    return tens


def make_grad_safe(network):
    for p in network.parameters():
        if p.grad is not None:
            p.grad.data.copy_(make_safe(p.grad))


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


def ca_seed(batch_size, n_filter, image_size, device, seed_value=1.0):
    # The canvas
    seed = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    # The starting point of the wave
    seed[:, 0, image_size // 2, image_size // 2] = seed_value
    return seed


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

