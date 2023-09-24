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
from src.distributions import get_zdist, load_multigauss_params, save_multigauss_params, update_multigauss_params
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

image_size = config['data']['image_size']
channels = config['data']['channels']
n_labels = config['data']['n_labels']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
lat_size = config['network']['kwargs']['lat_size']
injected_encoder = config['network']['kwargs'].get('injected_encoder', False)
letter_encoding = config['network']['kwargs']['letter_encoding']
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
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder' if injected_encoder else 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
if letter_encoding:
    networks_dict.update({
        'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoderS'},
        'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoderS'},
    })

model_manager = ModelManager('regae_ra', networks_dict, config)  #, to_avg=['encoder', 'decoder'])
encoder = model_manager.get_network('encoder')
# encoder_avg = model_manager.get_network('encoder', avg=True)
decoder = model_manager.get_network('decoder')
# decoder_avg = model_manager.get_network('decoder', avg=True)
if letter_encoding:
    letter_encoder = model_manager.get_network('letter_encoder')
    letter_decoder = model_manager.get_network('letter_decoder')

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
def noise_factor(call_idx, trainstep=0, ministeps=1000, reverse=False):
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


# # Distributions
# zdist_mu, zdist_cov = load_multigauss_params(config['training']['out_dir'] + '/regae_ra', lat_size, device=device)
# zdist_cov_inv = None
# zdist = get_zdist('multigauss', lat_size, device=device, mu=zdist_mu, cov=zdist_cov)
# zdist_momentum = 1 - 1e-3

noise_mask = None


def sample_regae(images_or_lat, labels, prev_perm=None, next_perm=None, dec_only=False):
    global noise_mask
    # global zdist_cov_inv

    if not dec_only:
        images = images_or_lat

        if injected_encoder:
            lat_enc, out_embs, _ = encoder(images, labels)
        else:
            lat_enc, out_embs, _ = encoder(images)

        if letter_encoding:
            letters = letter_encoder(lat_enc)
            # Latent space interpolation
            if prev_perm is not None:
                if noise_mask is None:
                    noise_mask = torch.rand([letters.size(0), 1, letters.size(2)], device=letters.device)
                letters = torch.where(noise_mask > fussion_progress, letters[prev_perm], letters[next_perm])
            else:
                noise_mask = None
            lat_dec = letter_decoder(letters)
        else:
            lat_dec = lat_enc

            # zdist_cov_inv = torch.pinverse(zdist_cov) if zdist_cov_inv is None else zdist_cov_inv
            # lat_dec_n = (lat_dec - zdist_mu) @ zdist_cov_inv

            # Latent space interpolation
            if prev_perm is not None:
                lat_dec = (1 - fussion_progress) * lat_dec[prev_perm] + fussion_progress * lat_dec[next_perm]

                if fussion_progress > 0.5:
                    labels = labels[next_perm]
                else:
                    labels = labels[prev_perm]

            # lat_dec = (lat_dec_n @ zdist_cov) + zdist_mu
    else:
        lat_dec = images_or_lat

    images_dec = torch.group_norm(torch.randn((batch_split_size, channels, image_size, image_size), device=device), 1)
    for call_idx in range(n_calls):
        if call_idx == 0:
            images_dec = decoder(lat_dec, images_dec, labels)
        elif call_idx == n_calls - 1:
            images_dec = decoder(None, images_dec, labels)
        else:
            images_dec = 0.5 * (decoder(lat_dec, images_dec, labels) + decoder(None, images_dec, labels))

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

window_size = math.ceil((len(trainloader) // batch_split) / 10)

train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
n_frames = video_fps * (train_phase + 1)
jamm_idcs = [i for i in range(1, batch_split_size)] + [0]

print('Starting training, phase: %d' % train_phase)

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split  # * (1 + train_phase)

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']), dynamic_ncols=True)
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                # loss_kl_sum = 0
                loss_dec_sum = 0
                loss_denoise_sum = 0

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                fussion_progress = ((model_manager.it // config['training']['stream_every']) % n_frames) / n_frames

                with model_manager.on_step(['encoder', 'decoder'] + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])) as nets_to_train:

                    for b in range(batch_mult):
                        ini = (b * batch_split_size) % batch_size
                        end = ini + batch_split_size
                        call_idcs = call_idcs_test[ini:end]
                        ministep = torch.randint(1000, (batch_split_size, ), device=device)
                        # np.random.shuffle(train_perm)

                        images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                        # Encoding
                        if injected_encoder:
                            lat_enc, _, _ = encoder(images, labels)
                        else:
                            lat_enc, _, _ = encoder(images)

                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            noise_mask = torch.rand([letters.size(0), 1, letters.size(2)], device=device)
                            rand_letters = torch.softmax(torch.randn_like(letters) * 10., dim=1)
                            letters = torch.where(noise_mask < (1 - 0.1 * min(train_phase + 1, 9)), letters, rand_letters)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc

                        # lat_dec = F.normalize(lat_dec)
                        #
                        # loss_kl = (1 / batch_mult) * 2e-3 * 1e-1 ** train_phase * age_gaussian_kl_loss(lat_dec)
                        # model_manager.loss_backward(loss_kl, nets_to_train, retain_graph=True)
                        # loss_kl_sum += loss_kl.item()

                        with torch.no_grad():
                            # zdist, zdist_mu, zdist_cov = update_multigauss_params(lat_size, zdist, zdist_mu, zdist_cov, lat_dec, zdist_momentum)
                            # zdist_cov_inv = None

                            nf = noise_factor(call_idcs, ministep)[:, None, None, None]
                            nf_next = noise_factor(call_idcs + 1, ministep)[:, None, None, None]
                            images_init = (1 - nf) * images + nf * torch.group_norm(torch.randn_like(images), 1)
                            images_target = (1 - nf_next) * images + nf_next * torch.group_norm(torch.randn_like(images), 1)

                        images_init.requires_grad_()

                        images_out = decoder(lat_dec, images_init, labels)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_out, images_target)
                        model_manager.loss_backward(loss_dec, nets_to_train)
                        loss_dec_sum += loss_dec.item()

                        with torch.no_grad():
                            nf = noise_factor(call_idcs, ministep, reverse=True)[:, None, None, None]
                            nf_next = noise_factor(call_idcs + 1, ministep, reverse=True)[:, None, None, None]
                            images_init = (1 - nf) * images + nf * torch.group_norm(torch.randn_like(images), 1)
                            images_target = (1 - nf_next) * images + nf_next * torch.group_norm(torch.randn_like(images), 1)

                        images_init.requires_grad_(True)

                        images_out = decoder(None, images_init, labels)

                        loss_denoise = (1 / batch_mult) * F.mse_loss(images_out, images_target)
                        model_manager.loss_backward(loss_denoise, nets_to_train)
                        loss_denoise_sum += loss_denoise.item()

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

                    stream_images(images_dec, config_name + '/regae_ra', config['training']['out_dir'] + '/regae_ra')

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
                model_manager.log_scalar('losses', 'loss_denoise', loss_denoise_sum)

    # save_multigauss_params(zdist_mu, zdist_cov, config['training']['out_dir'] + '/regae_ra')

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and (model_manager.epoch % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, trainiter, _ = get_inputs(trainiter, batch_size, device)
            images_fix_l = []
            images_dec_l = []
            images_gen_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size

                images_fix = sample_regae(images_test[ini:end], labels_test[ini:end])
                images_fix = torch.cat([images_test[ini:end], images_fix], dim=3)
                images_fix_l.append(images_fix)

                images_dec = sample_regae(images[ini:end], labels[ini:end])
                images_dec = torch.cat([images[ini:end], images_dec], dim=3)
                images_dec_l.append(images_dec)

                # images_gen = sample_regae(torch.randn((batch_split_size, lat_size), device=device), None, dec_only=True)
                # images_zgen = sample_regae(None, None, dec_only=True)
                # images_gen = torch.cat([images_gen, images_zgen], dim=3)
                # images_gen_l.append(images_gen)
            images_fix = torch.cat(images_fix_l)
            images_dec = torch.cat(images_dec_l)
            # images_gen = torch.cat(images_gen_l)
            model_manager.log_images(images_fix,  'fixed_samples')
            model_manager.log_images(images_dec,  'random_samples')
            # model_manager.log_images(images_gen,  'generated_samples')
            # for lab in range(config['training']['sample_labels']):
            #     if labels_test.dim() == 1:
            #         fixed_lab = torch.full((batch_split_size,), lab, device=device, dtype=torch.int64)
            #     else:
            #         fixed_lab = labels_test[ini:end].clone()
            #         fixed_lab[:, lab] = 1
            #     lat_gen, _, _ = encoder(images_test[ini:end], fixed_lab)
            #     images_gen = decoder(lat_gen, img_init=decoder(lat_gen)[0])[0]
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