# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import rand_circle_masks, tensor_jamm, mask_jamm, mask_templates
from src.utils.loss_utils import compute_gan_loss, update_reg_params, compute_grad_reg, compute_dir_grad_reg
from src.utils.model_utils import compute_inception_score, update_network_average
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images, video_fps
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a REGAE mon')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

config['training']['optimizer'] = 'adam05'

image_size = config['data']['image_size']
channels = config['data']['channels']
n_labels = config['data']['n_labels']
n_filter = config['network']['kwargs']['n_filter']
# n_calls = config['network']['kwargs']['n_calls']
lat_size = config['network']['kwargs']['lat_size']
n_epochs = config['training']['n_epochs']
d_reg_param = config['training']['d_reg_param']
d_reg_every = config['training']['d_reg_every']
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
    'encoder': {'class': config['network']['class'], 'sub_class':  'LabsInjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'ContrastiveDiscriminator'},
}

model_manager = ModelManager('regae_jamm', networks_dict, config) #, to_avg=['encoder', 'decoder'])
encoder = model_manager.get_network('encoder')
# encoder_avg = model_manager.get_network('encoder', avg=True)
decoder = model_manager.get_network('decoder')
# decoder_avg = model_manager.get_network('decoder', avg=True)
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

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
# def noise_factor(call_idx, trainstep=0, ministeps=1000, reverse=True):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     ministep = (trainstep % ministeps) / ministeps
#     if reverse:
#         return (1 - (1 / 1.5 ** (n_calls - call_idx - ministep))).clamp_(0, 1)
#     else:
#         return (1 / 1.5 ** (call_idx + ministep)).clamp_(0, 1)

# Noisy noise factor
# def noise_factor(call_idx):
#     if isinstance(call_idx, int):
#         call_idx = torch.ones([], device=device) * call_idx
#     return (n_calls - call_idx + 0.1 * np.random.randn()) / n_calls


def requantize(x):
    qx = x.clamp_(-1, 1).add_(1).mul_(255 / 2).int().clamp_(0, 255)
    return qx.type_as(x).mul_(2 / 255).add_(-1)


noise_init_sample = torch.group_norm(torch.randn((batch_split_size, channels, image_size, image_size), device=device), 1).detach_().requires_grad_(False)


def sample_regae(images, labels, prev_perm=None, next_perm=None):
    lat_dec, _, _ = encoder(images, labels)
    # Latent space interpolation
    if prev_perm is not None:
        lat_dec = (1 - fussion_progress) * lat_dec[prev_perm] + fussion_progress * lat_dec[next_perm]
    images_dec = decoder(noise_init_sample, lat_dec)

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
    # labels = (labels + 0.1 * torch.randn_like(labels)).clamp_(0, 1)
    images.requires_grad_()
    labels.requires_grad_()
    return images, labels, trainiter, idxes


# call_idcs_test = torch.arange(batch_size, device=device) % n_calls
images_test, labels_test, trainiter, _ = get_inputs(iter(trainloader), batch_size, device)

base_perm = np.arange(0, batch_split_size)
prev_perm = base_perm.copy()
next_perm = None

if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, trainiter, _ = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()

window_size = math.ceil((len(trainloader) // batch_split) / 10)

train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
n_frames = video_fps * (train_phase + 1)
jamm_idcs = [i for i in range(1, batch_split_size)] + [0]
mtemps = mask_templates(image_size * 2)

# # Discriminator reg target
reg_dis_target = (lr ** 2) * (0.7 ** (train_phase % 8))

d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', d_reg_every if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)

if train_phase > 1:
    new_sample_weights = trainloader.sampler.weights.clone()

torch.autograd.set_detect_anomaly(True)
print('Starting training, phase: %d' % train_phase)

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split * (1 + train_phase)

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            if train_phase > 1 and model_manager.it % 100 == 99:
                trainloader.sampler.weights = new_sample_weights
                with torch.no_grad():
                    new_sample_weights = new_sample_weights.clone()
                    new_sample_weights = (new_sample_weights + new_sample_weights.mean()) / 2

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum = 0, 0
                labs_dis_enc_sign, labs_dis_dec_sign = 0, 0
                loss_gen_enc_sum, loss_gen_dec_sum = 0, 0

                reg_dis_enc_sum, reg_dis_dec_sum = 0, 0

                if d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0:
                    d_reg_factor = (d_reg_every_mean_next - (model_manager.it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                else:
                    reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                    reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                fussion_progress = ((model_manager.it // config['training']['stream_every']) % n_frames) / n_frames

                with model_manager.on_step(['dis_encoder', 'discriminator']) as nets_to_train:

                    for b in range(batch_mult):
                        images, labels, trainiter, idxes = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            noise_init = torch.group_norm(torch.randn_like(images), 1)
                            images_init, idx = mask_jamm(images, noise_init, mtemps, scale_factor=2, return_idx=True, mask_weight=torch.rand((batch_split_size, 1, 1, 1), device=device))
                            lat_dec, _, _ = encoder(images_init, labels)
                            images_dec = decoder(noise_init, lat_dec)
                            images_dec = mask_jamm(images, images_dec, mtemps, scale_factor=2, idx=idx)

                        images_dec.requires_grad_()
                        lat_top_enc, out_embs_enc, _ = dis_encoder(images, labels)
                        lat_top_enc_c, _, _ = dis_encoder(images_dec, labels)

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

                        labs_enc = discriminator(lat_top_enc, lat_top_enc_c)
                        labs_dis_enc_sign += ((1 / batch_mult) * labs_enc.sign().mean()).item()

                        loss_dis_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 1)
                        model_manager.loss_backward(loss_dis_enc, nets_to_train, retain_graph=d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        if d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0:
                            reg_dis_enc = compute_grad_reg(loss_dis_enc, [images, images_dec])
                            model_manager.loss_backward(reg_dis_enc, nets_to_train)#, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item()

                            # if np.random.rand() >= 0.5:
                            #     labs_enc_c = discriminator(lat_top_enc_c, lat_top_enc)
                            #     loss_dis_enc_inv = (1 / batch_mult) * compute_gan_loss(labs_enc_c, 0)
                            # else:
                            #     loss_dis_enc_inv = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)

                            # labs_enc_c = discriminator(lat_top_enc_c, lat_top_enc)
                            # loss_dis_enc_c = (1 / batch_mult) * compute_gan_loss(labs_enc_c, 0)
                            #
                            # reg_dis_enc_dir = d_reg_factor * compute_dir_grad_reg(loss_dis_enc, loss_dis_enc_c, [images, images_dec])
                            # model_manager.loss_backward(reg_dis_enc_dir, nets_to_train)
                            # reg_dis_enc_sum += reg_dis_enc_dir.item() / d_reg_factor

                        images, labels, trainiter, idxes = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            noise_init = torch.group_norm(torch.randn_like(images), 1)
                            images_init, idx = mask_jamm(images, noise_init, mtemps, scale_factor=2, return_idx=True, mask_weight=torch.rand((batch_split_size, 1, 1, 1), device=device))
                            lat_dec, _, _ = encoder(images_init, labels)
                            images_dec = decoder(noise_init, lat_dec)
                            images_dec = mask_jamm(images, images_dec, mtemps, scale_factor=2, idx=idx)

                        images_dec.requires_grad_()
                        lat_top_dec, out_embs_dec, _ = dis_encoder(images, labels)
                        lat_top_dec_c, _, _ = dis_encoder(images_dec, labels)

                        if train_phase > 1:
                            with torch.no_grad():
                                inv_acts = 0.
                                for out_emb in out_embs_dec:
                                    inv_acts_max = out_emb.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                                    inv_acts_min = out_emb.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                                    inv_acts_tmp = (out_emb - inv_acts_min) / (inv_acts_max - inv_acts_min)
                                    inv_acts += 1 / (inv_acts_tmp.mean(dim=(1, 2, 3)) + 1e-4).to('cpu')
                                inv_acts /= len(out_embs_dec)
                                new_sample_weights[idxes] = (new_sample_weights[idxes] + inv_acts) / 2

                        labs_dec = discriminator(lat_top_dec_c, lat_top_dec)
                        labs_dis_dec_sign -= ((1 / batch_mult) * labs_dec.sign().mean()).item()

                        loss_dis_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 0)
                        model_manager.loss_backward(loss_dis_dec, nets_to_train, retain_graph=d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0)
                        loss_dis_dec_sum += loss_dis_dec.item()

                        if d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0:
                            # reg_dis_dec = compute_grad_reg(loss_dis_dec, [images, images_dec])
                            # model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            # reg_dis_dec_sum += reg_dis_dec.item()

                            # if np.random.rand() >= 0.5:
                            #     labs_dec_c = discriminator(lat_top_dec, lat_top_dec_c)
                            #     loss_dis_dec_inv = (1 / batch_mult) * compute_gan_loss(labs_dec_c, 1)
                            # else:
                            #     loss_dis_dec_inv = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)

                            labs_dec_c = discriminator(lat_top_dec, lat_top_dec_c)
                            loss_dis_dec_c = (1 / batch_mult) * compute_gan_loss(labs_dec_c, 1)

                            reg_dis_dec_dir = d_reg_factor * compute_dir_grad_reg(loss_dis_dec, loss_dis_dec_c, [images, images_dec])
                            model_manager.loss_backward(reg_dis_dec_dir, nets_to_train)
                            reg_dis_dec_sum += reg_dis_dec_dir.item() / d_reg_factor

                    if d_reg_every_mean > 0 and model_manager.it % int(d_reg_every_mean) == 0:
                        reg_dis_max = reg_dis_dec_sum  # max(reg_dis_enc_sum, reg_dis_dec_sum)
                        loss_dis_min = min(loss_dis_enc_sum, loss_dis_dec_sum)
                        d_reg_every_mean = d_reg_every_mean_next
                        d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every,
                                                                                    d_reg_param_mean, 1 / d_reg_param,
                                                                                    reg_dis_max, reg_dis_target,
                                                                                    loss_dis_min)

                with model_manager.on_step(['encoder', 'decoder']) as nets_to_train:

                    for b in range(batch_mult):
                        # images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)
                        #
                        # with torch.no_grad():
                        #     noise_init = torch.group_norm(torch.randn_like(images), 1)
                        #
                        # images_init, idx = mask_jamm(images, noise_init, mtemps, scale_factor=2, return_idx=True, mask_weight=torch.rand((batch_split_size, 1, 1, 1), device=device))
                        # lat_dec, _, _ = encoder(images_init, labels)
                        # images_dec = decoder(noise_init, lat_dec)
                        # images_dec = mask_jamm(images, images_dec, mtemps, scale_factor=2, idx=idx)
                        #
                        # lat_top_enc, _, _ = dis_encoder(images, labels)
                        # lat_top_enc_c, _, _ = dis_encoder(images_dec, labels)
                        # labs_enc = discriminator(lat_top_enc, lat_top_enc_c)
                        #
                        # loss_gen_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)
                        # model_manager.loss_backward(loss_gen_enc, nets_to_train)
                        # loss_gen_enc_sum += loss_gen_enc.item()

                        images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            noise_init = torch.group_norm(torch.randn_like(images), 1)

                        images_init, idx = mask_jamm(images, noise_init, mtemps, scale_factor=2, return_idx=True, mask_weight=torch.rand((batch_split_size, 1, 1, 1), device=device))
                        lat_dec, _, _ = encoder(images_init, labels)
                        images_dec = decoder(noise_init, lat_dec)
                        images_dec = mask_jamm(images, images_dec, mtemps, scale_factor=2, idx=idx)

                        lat_top_dec, _, _ = dis_encoder(images, labels)
                        lat_top_dec_c, _, _ = dis_encoder(images_dec, labels)
                        labs_dec = discriminator(lat_top_dec_c, lat_top_dec)

                        loss_gen_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                        model_manager.loss_backward(loss_gen_dec, nets_to_train)
                        loss_gen_dec_sum += loss_gen_dec.item()

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

                    stream_images(images_dec, config_name + '/regae_jamm', config['training']['out_dir'] + '/regae_jamm')

                if ((model_manager.it + 1) / config['training']['stream_every']) % n_frames == 0:
                    prev_perm = next_perm
                if ((model_manager.it + 1) / config['training']['stream_every']) % (2 * n_frames) == 0:
                    next_perm = None

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_enc_sum + loss_gen_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dis='%.2e' % (np.sum(running_loss_dis) / running_factor),
                              loss_gen='%.2e' % (np.sum(running_loss_gen) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                model_manager.log_scalar('losses', 'loss_dis_enc',  loss_dis_enc_sum)
                model_manager.log_scalar('losses', 'labs_dis_enc_sign',  labs_dis_enc_sign)
                model_manager.log_scalar('losses', 'loss_dis_dec',  loss_dis_dec_sum)
                model_manager.log_scalar('losses', 'labs_dis_dec_sign',  labs_dis_dec_sign)

                # model_manager.log_scalar('losses', 'loss_gen_enc', loss_gen_enc_sum)
                model_manager.log_scalar('losses', 'loss_gen_dec', loss_gen_dec_sum)

                model_manager.log_scalar('regs', 'reg_dis_enc',  reg_dis_enc_sum)
                model_manager.log_scalar('regs', 'reg_dis_dec',  reg_dis_dec_sum)
                model_manager.log_scalar('regs', 'd_reg_every_mean',  d_reg_every_mean)
                model_manager.log_scalar('regs', 'd_reg_param_mean',  d_reg_param_mean)

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
