# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import rand_change_letters, mask_templates
from src.utils.model_utils import grad_ema_update
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images, video_fps
from os.path import basename, splitext
import sys
import select

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train D REGAE mon!')
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
n_samples = int(n_calls ** 0.5)
injected_encoder = config['network']['kwargs'].get('injected_encoder', False)
letter_encoding = config['network']['kwargs']['letter_encoding']
n_epochs = config['training']['n_epochs']
batch_size = config['training']['batch_size']
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
    'encoder': {'class': config['network']['class'], 'sub_class': 'NInjectedEncoder' if injected_encoder else 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
if letter_encoding:
    networks_dict.update({
        'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoder'},
        'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoder'},
    })

model_manager = ModelManager('dregae', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
if letter_encoding:
    letter_encoder = model_manager.get_network('letter_encoder')
    letter_decoder = model_manager.get_network('letter_decoder')

model_manager.print()

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


images_test, labels_test, trainiter, _ = get_inputs(iter(trainloader), batch_size, device)

base_perm = np.arange(0, batch_split_size)
prev_perm = base_perm.copy()
next_perm = None

window_size = math.ceil((len(trainloader) // batch_split) / 10)

train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
print('Starting training, phase: %d' % train_phase)

mtemps = mask_templates(image_size * 2)

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split * (1 + train_phase)

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']), dynamic_ncols=True)
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_enc_sum, loss_denoise_sum = 0, 0

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                fussion_progress = (model_manager.it % video_fps) / video_fps

                with model_manager.on_step(['encoder']) as nets_to_train:

                    for b in range(batch_mult):

                        for s in range(n_samples):

                            images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                            # Encoding
                            if injected_encoder:
                                lat_enc, _, _ = encoder(images, labels)
                            else:
                                lat_enc, _, _ = encoder(images)

                            loss_enc = (1 / (batch_mult * n_samples)) * F.mse_loss(lat_enc, torch.zeros_like(lat_enc))
                            model_manager.loss_backward(loss_enc, nets_to_train)
                            loss_enc_sum += loss_enc.item()

                with model_manager.on_step(['decoder'] + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])) as nets_to_train:

                    for b in range(batch_mult):

                        for s in range(n_samples):

                            images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                            # Encoding
                            with torch.no_grad():
                                if injected_encoder:
                                    lat_enc, _, _ = encoder(images, labels)
                                else:
                                    lat_enc, _, _ = encoder(images)

                            lat_enc.requires_grad_()

                            if letter_encoding:
                                letters = letter_encoder(lat_enc)
                                letters = rand_change_letters(letters)
                                lat_dec = letter_decoder(letters)
                            else:
                                lat_dec = lat_enc

                            if (model_manager.it + b + s) % 3 == 0:
                                noise_inits = [torch.randn_like(images)]
                            elif (model_manager.it + b + s) % 3 == 1:
                                scale_factor = 1 / 2 ** np.random.randint(int(np.log2(image_size)) - 1)
                                if scale_factor < 1:
                                    resized_images = F.interpolate(images, scale_factor=scale_factor, mode='bilinear', align_corners=False)
                                else:
                                    resized_images = images
                                b, c, h, w = resized_images.size()
                                noise_inits = resized_images.reshape(b, c, h * w)[:, :, np.random.choice(np.arange(0, h * w), h * w, replace=False)].reshape(b, c, h, w)
                                noise_inits = torch.randn_like(resized_images) * (noise_inits - resized_images)
                                if scale_factor < 1:
                                    noise_inits = F.interpolate(noise_inits, size=image_size, mode='bilinear', align_corners=False)
                                noise_inits = [noise_inits.detach_()]
                            elif (model_manager.it + b + s) % 3 == 2:
                                noise_inits = torch.empty_like(images)
                                torch.nn.init.orthogonal_(noise_inits)
                                noise_inits = [(noise_inits / noise_inits.std())]

                            call_init = np.random.randint(0, n_calls - 1)

                            _, _, noise_outs = decoder(lat_dec, images, noise_inits, call_init, call_init + 1)

                            loss_denoise = (1 / (batch_mult * n_samples)) * F.mse_loss(torch.cat(noise_outs), torch.cat(noise_inits))
                            model_manager.loss_backward(loss_denoise, nets_to_train)
                            loss_denoise_sum += loss_denoise.item()

                    # grad_ema_update(encoder)
                    # grad_ema_update(decoder)
                    # if letter_encoding:
                    #     grad_ema_update(letter_encoder)
                    #     grad_ema_update(letter_decoder)

                # Streaming Images
                with torch.no_grad():
                    images_dec_l = []
                    for i in range(batch_split):
                        ini = i * batch_split_size
                        end = (i + 1) * batch_split_size
                        if injected_encoder:
                            lat_enc, out_embs, _ = encoder(images_test[ini:end], labels_test[ini:end])
                        else:
                            lat_enc, out_embs, _ = encoder(images_test[ini:end])

                        # Pre-Latent space interpolation
                        # out_emb = out_embs[-1]
                        # out_emb = (1 - fussion_progress) * out_emb[prev_perm] + fussion_progress * out_emb[next_perm]
                        # lat_enc = encoder.out_to_lat(out_emb.reshape(batch_split_size, -1))

                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc

                        # Latent space interpolation
                        lat_dec = (1 - fussion_progress) * lat_dec[prev_perm] + fussion_progress * lat_dec[next_perm]
                        images_dec = decoder(lat_dec)[0]

                        # Feature space interpolation
                        # if fussion_progress > 0. and (prev_perm != next_perm).any():
                        #     images_dec = decoder.mix_forward(lat_dec[prev_perm], lat_dec[next_perm], (1 - fussion_progress))
                        # else:
                        #     images_dec = decoder(lat_dec[prev_perm])[0]

                        images_dec_l.append(images_dec)
                    images_dec = torch.cat(images_dec_l)

                stream_images(images_dec, config_name + '/dregae', config['training']['out_dir'] + '/dregae')

                if (model_manager.it + 1) % video_fps == 0:
                    prev_perm = next_perm
                if (model_manager.it + 1) % (2 * video_fps) == 0:
                    next_perm = None

                # Print progress
                running_loss[batch % window_size] = loss_denoise_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                model_manager.log_scalar('losses', 'loss_enc', loss_enc_sum)
                model_manager.log_scalar('losses',  'loss_denoise',  loss_denoise_sum)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and (model_manager.epoch % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, trainiter, _ = get_inputs(trainiter, batch_size, device)
            images_gen_l = []
            images_dec_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size

                lat_gen, _, _ = encoder(images_test[ini:end], labels_test[ini:end])
                images_gen = decoder(lat_gen)[0]
                # images_gen_avg = decoder_avg(lat_gen, img_init=decoder_avg(lat_gen)[0])[0]
                images_gen = torch.cat([images_test[ini:end], images_gen], dim=3)
                images_gen_l.append(images_gen)

                lat_enc, _, _ = encoder(images[ini:end], labels[ini:end])
                images_dec = decoder(lat_enc)[0]
                # images_dec_avg = decoder_avg(lat_enc, img_init=decoder_avg(lat_enc)[0])[0]
                images_dec = torch.cat([images[ini:end], images_dec], dim=3)
                images_dec_l.append(images_dec)
            images_gen = torch.cat(images_gen_l)
            images_dec = torch.cat(images_dec_l)
            model_manager.log_images(images_gen,  'fixed_samples')
            model_manager.log_images(images_dec,  'random_samples')
            # for lab in range(config['training']['sample_labels']):
            #     if labels_test.dim() == 1:
            #         fixed_lab = torch.full((batch_split_size,), lab, device=device, dtype=torch.int64)
            #     else:
            #         fixed_lab = labels_test[ini:end].clone()
            #         fixed_lab[:, lab] = 1
            #     lat_gen, _, _ = encoder_avg(images_test[ini:end], fixed_lab)
            #     images_gen = decoder(lat_gen, img_init=decoder(lat_gen)[0])[0]
            #     # images_gen_avg = decoder_avg(lat_gen, img_init=decoder_avg(lat_gen)[0])[0]
            #     images_gen = torch.cat([images_gen], dim=3)
            #     model_manager.log_images(images_gen,  'class_%04d' % lab)

        # # Perform inception
        # if config['training']['inception_every'] > 0 and (model_manager.epoch % config['training']['inception_every']) == 0 and model_manager.epoch > 0:
        #     t.write('Computing inception/fid!')
        #     inception_mean, inception_std, fid = compute_inception_score(generator, decoder_avg,
        #                                                                  10000, 10000, config['training']['batch_size'],
        #                                                                  zdist, ydist, fid_real_samples, device)
        #     model_manager.log_scalar('inception_score',  'mean',  inception_mean)
        #     model_manager.log_scalar('inception_score',  'stddev',  inception_std)
        #     model_manager.log_scalar('inception_score',  'fid',  fid)

model_manager.save()
print('Training is complete...')
