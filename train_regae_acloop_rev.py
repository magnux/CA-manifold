# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.loss_utils import age_gaussian_kl_loss
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images, video_fps
from os.path import basename, splitext
import sys
import select

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(420)
torch.manual_seed(420)

parser = argparse.ArgumentParser(description='Train a REGAE mon!')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

config['network']['kwargs']['n_conds'] = 2
config['network']['kwargs']['reversible'] = True

image_size = config['data']['image_size']
channels = config['data']['channels']
n_labels = config['data']['n_labels']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
lat_size = config['network']['kwargs']['lat_size']
# letter_encoding = config['network']['kwargs']['letter_encoding']
n_epochs = config['training']['n_epochs']
batch_size = config['training']['batch_size']
lr = config['training']['lr']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
batch_mult_steps = config['training']['batch_mult_steps']
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
samples_weight = torch.ones(len(trainset)) / len(trainset)
sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=False, num_workers=n_workers, drop_last=True, sampler=sampler)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Networks
networks_dict = {
    'clean_encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'noisy_encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
# if letter_encoding:
#     networks_dict.update({
#         'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoderS'},
#         'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoderS'},
#     })

model_manager = ModelManager('regae_acloop_rev', networks_dict, config, to_avg=['clean_encoder', 'noisy_encoder', 'decoder'])
clean_encoder = model_manager.get_network('clean_encoder')
clean_encoder_avg = model_manager.get_network('clean_encoder', avg=True)
noisy_encoder = model_manager.get_network('noisy_encoder')
noisy_encoder_avg = model_manager.get_network('noisy_encoder', avg=True)
decoder = model_manager.get_network('decoder')
decoder_avg = model_manager.get_network('decoder', avg=True)
# if letter_encoding:
#     letter_encoder = model_manager.get_network('letter_encoder')
#     letter_decoder = model_manager.get_network('letter_decoder')

model_manager.print()


# Positive Cosine noise factor
# def noise_factor(call_idx):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     return 0.5 * (torch.cos(np.pi * call_idx / n_calls) + 1)

# Cosine noise factor
# def noise_factor(call_idx):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     return torch.cos(np.pi * call_idx / (n_calls * 2))

# Linear noise factor
# def noise_factor(call_idx, n_calls):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     return (n_calls - call_idx) / n_calls

# Moving linear noise factor
# def noise_factor(call_idx, trainstep=0, ministeps=1000):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     ministep = (trainstep % ministeps) / ministeps
#     return (n_calls - call_idx - ministep).clamp_(0, n_calls) / n_calls

# Fractal (p-addic) noise factor
def noise_factor(call_idx, trainstep=0, ministeps=1000, reverse=True):
    if isinstance(call_idx, int):
        call_idx = torch.ones([], device=device) * call_idx
    ministep = (trainstep % ministeps) / ministeps
    if reverse:
        return (1 - (1 / 1.5 ** (n_calls - call_idx - ministep))).clamp_(0, 1)
    else:
        return (1 / 1.5 ** (call_idx + ministep)).clamp_(0, 1)

# Noisy noise factor
# def noise_factor(call_idx):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     return (n_calls - call_idx + 0.1 * np.random.randn()) / n_calls


def requantize(x):
    qx = x.clamp_(-1, 1).add_(1).mul_(255 / 2).int().clamp_(0, 255)
    return qx.type_as(x).mul_(2 / 255).add_(-1)


noise_mask = None
noise_init_sample = torch.group_norm(torch.randn((batch_split_size, channels, image_size, image_size), device=device), 1).detach_().requires_grad_(False)


def sample_regae(images_or_lat, labels, prev_perm=None, next_perm=None, dec_only=False, use_avg=True):
    global noise_mask
    # global zdist_cov_inv

    if use_avg:
        clean_encoder_sample = clean_encoder_avg
        noisy_encoder_sample = noisy_encoder_avg
        decoder_sample = decoder_avg
    else:
        clean_encoder_sample = clean_encoder
        noisy_encoder_sample = noisy_encoder
        decoder_sample = decoder

    if not dec_only:
        images = images_or_lat

        lat_enc, out_embs, _ = clean_encoder_sample(images, (labels[next_perm] if fussion_progress > 0.5 else labels[prev_perm]) if prev_perm is not None else labels)

        # if letter_encoding:
        #     letters = letter_encoder(lat_enc)
        #     # Latent space interpolation
        #     if prev_perm is not None:
        #         if noise_mask is None:
        #             noise_mask = torch.rand([letters.size(0), 1, letters.size(2)], device=letters.device)
        #         letters = torch.where(noise_mask > fussion_progress, letters[prev_perm], letters[next_perm])
        #     else:
        #         noise_mask = None
        #     lat_dec = letter_decoder(letters)
        # else:
        lat_dec = lat_enc

        # zdist_cov_inv = torch.pinverse(zdist_cov) if zdist_cov_inv is None else zdist_cov_inv
        # lat_dec_n = (lat_dec - zdist_mu) @ zdist_cov_inv

        # Latent space interpolation
        if prev_perm is not None:
            lat_dec = (1 - fussion_progress) * lat_dec[prev_perm] + fussion_progress * lat_dec[next_perm]

        # lat_dec = (lat_dec_n @ zdist_cov) + zdist_mu
    else:
        lat_dec = images_or_lat

    images_dec = decoder_sample(noise_init_sample, (lat_dec, None))

    for call_idx in range(n_calls):
        nf = noise_factor(call_idx)
        images_init = requantize(images_dec) + nf * torch.group_norm(torch.randn((batch_split_size, channels, image_size, image_size), device=device), 1).detach_().requires_grad_(False)
        nlat_dec, _, _ = noisy_encoder_sample(images_init, (labels[next_perm] if fussion_progress > 0.5 else labels[prev_perm]) if prev_perm is not None else labels)
        images_dec = decoder_sample(images_init, (None, nlat_dec))

    return requantize(images_dec)


embedding_mat = torch.eye(n_labels, device=device)


def get_inputs(trainiter, batch_size, device):
    images, labels, idxes = [], [], []
    n_batches = math.ceil(batch_size / batch_split_size)
    for _ in range(n_batches):
        next_inputs = next(trainiter, None)
        if trainiter is None or next_inputs is None:
            trainiter = iter(trainloader)
            next_inputs = next(trainiter, None)
        images.append(next_inputs[0])
        labels.append(next_inputs[1])
        idxes.extend(next_inputs[2].numpy().tolist())
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    if batch_size % config['training']['batch_size'] > 0:
        images, labels = images[:batch_size, ...], labels[:batch_size, ...]
        idxes = idxes[:batch_size]
    images, labels = images.to(device), labels.to(device)
    if labels.dtype is torch.int64:
        if labels.dim() == 1:
            labels = embedding_mat[labels]
        else:
            labels = labels.to(torch.float32)
    images.requires_grad_()
    labels.requires_grad_()
    return images, labels, trainiter, idxes


call_idcs_test = torch.arange(batch_size, device=device) % n_calls
images_test, labels_test, trainiter, _ = get_inputs(iter(trainloader), batch_size, device)

base_perm = np.arange(0, batch_split_size)
# train_perm = base_perm.copy()
prev_perm = base_perm.copy()
next_perm = None
rev_switch = torch.ones((batch_split_size, 1), device=device)

window_size = math.ceil((len(trainloader) // batch_split) / 10)

train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
n_frames = video_fps * (train_phase + 1)
jamm_idcs = [i for i in range(1, batch_split_size)] + [0]

if train_phase > 1:
    new_sample_weights = trainloader.sampler.weights.clone()

print('Starting training, phase: %d' % train_phase)

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split  # * (1 + train_phase)

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']), dynamic_ncols=True)
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            if train_phase > 1 and model_manager.it % 100 == 99:
                trainloader.sampler.weights = new_sample_weights
                with torch.no_grad():
                    new_sample_weights = new_sample_weights.clone()
                    new_sample_weights = (new_sample_weights + new_sample_weights.mean()) / 2

            with model_manager.on_batch():

                # loss_kl_sum = 0
                loss_dec_sum, loss_dec_rev_sum = 0, 0
                loss_denoise_sum, loss_denoise_rev_sum = 0, 0

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                fussion_progress = ((model_manager.it // config['training']['stream_every']) % n_frames) / n_frames

                with model_manager.on_step(['clean_encoder', 'noisy_encoder', 'decoder']) as nets_to_train:  # + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])

                    for b in range(batch_mult * 2):
                        # np.random.shuffle(train_perm)

                        images, labels, trainiter, idxes = get_inputs(trainiter, batch_split_size, device)

                        lat_enc, out_embs_enc, _ = clean_encoder(images, labels)

                        if train_phase > 1:
                            with torch.no_grad():
                                inv_acts = 0.
                                for out_emb in out_embs_enc:
                                    inv_acts_max = out_emb.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                                    inv_acts_min = out_emb.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                                    inv_acts_tmp = (out_emb - inv_acts_min) / (inv_acts_max - inv_acts_min)
                                    inv_acts += 1 / (inv_acts_tmp.mean(dim=(1, 2, 3)) + 1e-4).to('cpu')
                                inv_acts /= len(out_embs_enc)
                                new_sample_weights[idxes] = (new_sample_weights[idxes] + inv_acts) / 2

                        # if letter_encoding:
                        #     letters = letter_encoder(lat_enc)
                        #     noise_mask = torch.rand([letters.size(0), 1, letters.size(2)], device=device)
                        #     rand_letters = torch.softmax(torch.randn_like(letters) * 10., dim=1)
                        #     letters = torch.where(noise_mask < (1 - 0.1 * min(train_phase + 1, 9)), letters, rand_letters)
                        #     lat_dec = letter_decoder(letters)
                        # else:
                        lat_dec = lat_enc

                        # lat_dec = F.normalize(lat_dec)
                        #
                        # loss_kl = (1 / batch_mult) * 2e-3 * 1e-1 ** train_phase * age_gaussian_kl_loss(lat_dec)
                        # model_manager.loss_backward(loss_kl, nets_to_train, retain_graph=True)
                        # loss_kl_sum += loss_kl.item()

                        with torch.no_grad():
                            noise_init = torch.group_norm(torch.randn_like(images), 1)
                            ini = (b * batch_split_size) % batch_size
                            end = ini + batch_split_size
                            call_idcs = call_idcs_test[ini:end]
                            ministep = torch.randint(1000, (batch_split_size,), device=device)
                            nf = noise_factor(call_idcs, ministep)[:, None, None, None]

                        images_init = (1 - nf) * images + nf * noise_init

                        images_out = decoder(images_init, (lat_dec, None), rev_switch if b % 2 == 1 else None)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_out, noise_init if b % 2 == 1 else images)
                        model_manager.loss_backward(loss_dec, nets_to_train)
                        if b % 2 == 0:
                            loss_dec_sum += loss_dec.item()
                        else:
                            loss_dec_rev_sum += loss_dec.item()

                        images_init = images + nf * noise_init

                        nlat_dec, out_embs_enc, _ = noisy_encoder(images_init, labels)

                        if train_phase > 1:
                            with torch.no_grad():
                                inv_acts = 0.
                                for out_emb in out_embs_enc:
                                    inv_acts_max = out_emb.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                                    inv_acts_min = out_emb.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                                    inv_acts_tmp = (out_emb - inv_acts_min) / (inv_acts_max - inv_acts_min)
                                    inv_acts += 1 / (inv_acts_tmp.mean(dim=(1, 2, 3)) + 1e-4).to('cpu')
                                inv_acts /= len(out_embs_enc)
                                new_sample_weights[idxes] = (new_sample_weights[idxes] + inv_acts) / 2

                        images_out = decoder(images_init, (None, nlat_dec), rev_switch if b % 2 == 1 else None)

                        loss_denoise = (1 / batch_mult) * F.mse_loss(images_out, noise_init if b % 2 == 1 else images)
                        model_manager.loss_backward(loss_denoise, nets_to_train)
                        if b % 2 == 0:
                            loss_denoise_sum += loss_denoise.item()
                        else:
                            loss_denoise_rev_sum += loss_denoise.item()

                model_manager.set_err(loss_dec_sum + loss_denoise_sum)

                if config['training']['stream_every'] > 0 and (model_manager.it % config['training']['stream_every']) == 0:
                    # Streaming Images
                    with torch.no_grad():
                        images_dec_l = []
                        for i in range(batch_split):
                            ini = i * batch_split_size
                            end = (i + 1) * batch_split_size
                            images_dec = sample_regae(images_test[ini:end], labels_test[ini:end], prev_perm, next_perm)

                            images_dec_l.append(images_dec)
                        images_dec = torch.cat(images_dec_l)

                    stream_images(images_dec, config_name + '/regae_acloop_rev', config['training']['out_dir'] + '/regae_acloop_rev')

                if ((model_manager.it + 1) / config['training']['stream_every']) % n_frames == 0:
                    prev_perm = next_perm
                if ((model_manager.it + 1) / config['training']['stream_every']) % (2 * n_frames) == 0:
                    next_perm = None

                # Print progress
                running_loss[batch % window_size] = loss_dec_sum + loss_denoise_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                # model_manager.log_scalar('losses', 'loss_kl', loss_kl_sum)
                model_manager.log_scalar('losses', 'loss_dec', loss_dec_sum)
                model_manager.log_scalar('losses', 'loss_dec_rev', loss_dec_rev_sum)
                model_manager.log_scalar('losses', 'loss_denoise', loss_denoise_sum)
                model_manager.log_scalar('losses', 'loss_denoise_rev', loss_denoise_rev_sum)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and (model_manager.epoch % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, trainiter, _ = get_inputs(trainiter, batch_size, device)
            images_fix_l = []
            images_fix_curr_l = []
            images_dec_l = []
            images_gen_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size

                images_fix = sample_regae(images_test[ini:end], labels_test[ini:end])
                images_fix = torch.cat([images_test[ini:end], images_fix], dim=3)
                images_fix_l.append(images_fix)

                images_fix_curr = sample_regae(images_test[ini:end], labels_test[ini:end], use_avg=False)
                images_fix_curr = torch.cat([images_test[ini:end], images_fix_curr], dim=3)
                images_fix_curr_l.append(images_fix_curr)

                images_dec = sample_regae(images[ini:end], labels[ini:end])
                images_dec = torch.cat([images[ini:end], images_dec], dim=3)
                images_dec_l.append(images_dec)

                # images_gen = sample_regae(torch.randn((batch_split_size, lat_size), device=device), None, dec_only=True)
                # images_zgen = sample_regae(None, None, dec_only=True)
                # images_gen = torch.cat([images_gen, images_zgen], dim=3)
                # images_gen_l.append(images_gen)
            images_fix = torch.cat(images_fix_l)
            images_fix_curr = torch.cat(images_fix_curr_l)
            images_dec = torch.cat(images_dec_l)
            # images_gen = torch.cat(images_gen_l)
            model_manager.log_images(images_fix,  'fixed_samples')
            model_manager.log_images(images_fix_curr, 'fixed_current_samples')
            model_manager.log_images(images_dec,  'random_samples')
            # model_manager.log_images(images_gen,  'generated_samples')
            # for lab in range(config['training']['sample_labels']):
            #     if labels_test.dim() == 1:
            #         fixed_lab = torch.full((batch_split_size,), lab, device=device, dtype=torch.int64)
            #     else:
            #         fixed_lab = labels_test[ini:end].clone()
            #         fixed_lab[:, lab] = 1
            #     lat_gen, _, _ = encoder(images_test[ini:end], fixed_lab)
            #     images_gen = decoder(img_init=decoder(lat_gen)[0], lat_gen)[0]
            #     images_gen = torch.cat([images_gen], dim=3)
            #     model_manager.log_images(images_gen,  'class_%04d' % lab)

        # # Perform inception
        # if config['training']['inception_every'] > 0 and (model_manager.epoch % config['training']['inception_every']) == 0 and model_manager.epoch > 0:
        #     t.write('Computing inception/fid!')
        #     inception_mean, inception_std, fid = compute_inception_score(generator, decoder,
        #                                                                  10000, 10000, config['training']['batch_size'],
        #                                                                  zdist, ydist, fid_real_samples, device)
        #     model_manager.log_scalar('inception_score',  'mean',  inception_mean)
        #     model_manager.log_scalar('inception_score',  'stddev',  inception_std)
        #     model_manager.log_scalar('inception_score',  'fid',  fid)

model_manager.save()
print('Training is complete...')
