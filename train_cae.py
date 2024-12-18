# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.loss_utils import fft_mse_loss
from src.utils.media_utils import rand_erase_images, rand_change_letters, rand_circle_masks, mask_jamm, mask_templates
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images, video_fps
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a CAE')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_labels = config['data']['n_labels']
n_filter = config['network']['kwargs']['n_filter']
injected_encoder = config['network']['kwargs'].get('injected_encoder', False)
letter_encoding = config['network']['kwargs']['letter_encoding']
n_epochs = config['training']['n_epochs']
persistence = config['training']['persistence']
regeneration = config['training']['regeneration']
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
    'encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder' if injected_encoder else 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
if letter_encoding:
    networks_dict.update({
        'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoder'},
        'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoder'},
    })

model_manager = ModelManager('cae', networks_dict, config)
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

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum, loss_pers_sum, loss_regen_sum = 0, 0, 0

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                fussion_progress = (model_manager.it % video_fps) / video_fps

                with model_manager.on_step(['encoder', 'decoder'] + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])) as nets_to_train:

                    for _ in range(batch_mult):

                        images, labels, trainiter, _ = get_inputs(trainiter, batch_split_size, device)
                        init_samples = None

                        # Obscure the input
                        if model_manager.it % 2 == 1:
                            # images_jammed = rand_erase_images(images)
                            images_jammed = mask_jamm(images, torch.zeros_like(images), mtemps, scale_factor=2)
                        else:
                            images_jammed = images

                        # Encoding
                        if injected_encoder:
                            lat_enc, _, _ = encoder(images_jammed, labels)
                        else:
                            lat_enc, _, _ = encoder(images_jammed)

                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            letters = rand_change_letters(letters)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc #+ (1e-3 * torch.randn_like(lat_enc))

                        # Decoding
                        images_dec, out_embs, _ = decoder(lat_dec)

                        loss_dec = (1 / batch_mult) * fft_mse_loss(images_dec, images)
                        model_manager.loss_backward(loss_dec, nets_to_train, retain_graph=True if (persistence or regeneration) else False)
                        loss_dec_sum += loss_dec.item()

                        if persistence:
                            n_calls_save = model_manager.get_n_calls('decoder')

                            # pers_steps = n_calls_save // 4
                            # model_manager.set_n_calls('decoder', pers_steps)

                            _, pers_out_embs, _ = decoder(lat_dec, out_embs[-1])

                            pers_target_out_embs = torch.cat([out_embs[-1] for _ in range(n_calls_save)])
                            loss_pers = (1 / batch_mult) * fft_mse_loss(torch.cat(pers_out_embs[1:]), pers_target_out_embs)

                            # # Loss on image output
                            # if isinstance(decoder, torch.nn.DataParallel):
                            #     pers_redec_images = decoder.module.out_conv(torch.cat(pers_out_embs[1:]))
                            # else:
                            #     pers_redec_images = decoder.out_conv(torch.cat(pers_out_embs[1:]))
                            #
                            # pers_target_images = torch.cat([images for _ in range(n_calls_save)])
                            # loss_pers += (1 / batch_mult) * fft_mse_loss(pers_redec_images, pers_target_images)

                            model_manager.loss_backward(loss_pers, nets_to_train, retain_graph=True)
                            loss_pers_sum += loss_pers.item()

                            # model_manager.set_n_calls('decoder', n_calls_save)

                        if regeneration:
                            n_calls_save = model_manager.get_n_calls('decoder')

                            # regen_steps = n_calls_save // 4
                            # model_manager.set_n_calls('decoder', regen_steps)

                            corrupt_init = rand_circle_masks(out_embs[-1])[model_manager.it % 3]
                            regen_target_out_embs = torch.cat([out_embs[-1] for _ in range(n_calls_save)])
                            _, regen_out_embs, _ = decoder(lat_dec, corrupt_init)

                            loss_regen = (1 / batch_mult) * fft_mse_loss(torch.cat(regen_out_embs[1:]), regen_target_out_embs)

                            # # Loss on image output
                            # if isinstance(decoder, torch.nn.DataParallel):
                            #     regen_redec_images = decoder.module.out_conv(torch.cat(regen_out_embs[1:]))
                            # else:
                            #     regen_redec_images = decoder.out_conv(torch.cat(regen_out_embs[1:]))
                            #
                            # regen_target_images = torch.cat([images for _ in range(n_calls_save)])
                            # loss_regen += (1 / batch_mult) * fft_mse_loss(regen_redec_images, regen_target_images)

                            model_manager.loss_backward(loss_regen, nets_to_train)
                            loss_regen_sum += loss_regen.item()

                            # model_manager.set_n_calls('decoder', n_calls_save)

                # Streaming Images
                with torch.no_grad():
                    images_dec_l = []
                    for i in range(batch_split):
                        ini = i * batch_split_size
                        end = (i + 1) * batch_split_size
                        if injected_encoder:
                            lat_enc, _, _ = encoder(images_test[ini:end], labels_test[ini:end])
                        else:
                            lat_enc, _, _ = encoder(images_test[ini:end])
                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc
                        lat_dec = (1 - fussion_progress) * lat_dec[prev_perm] + fussion_progress * lat_dec[next_perm]
                        images_dec, _, _ = decoder(lat_dec)
                        images_dec_l.append(images_dec)
                    images_dec = torch.cat(images_dec_l)

                stream_images(images_dec, config_name + '/cae', config['training']['out_dir'] + '/cae')

                if (model_manager.it + 1) % video_fps == 0:
                    prev_perm = next_perm
                if (model_manager.it + 1) % (2 * video_fps) == 0:
                    next_perm = None

                # Print progress
                running_loss[batch % window_size] = loss_dec_sum + loss_pers_sum + loss_regen_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                model_manager.log_scalar('losses',  'loss_dec',  loss_dec_sum)
                if persistence:
                    model_manager.log_scalar('losses',  'loss_pers',  loss_pers_sum)
                if regeneration:
                    model_manager.log_scalar('losses',  'loss_regen',  loss_regen_sum)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((model_manager.epoch + 1) % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, trainiter, _ = get_inputs(trainiter, batch_size, device)

            images_dec_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size

                if injected_encoder:
                    lat_enc, _, _ = encoder(images[ini:end], labels[ini:end])
                else:
                    lat_enc, _, _ = encoder(labels[ini:end])
                if letter_encoding:
                    letters = letter_encoder(lat_enc)
                    lat_dec = letter_decoder(letters)
                else:
                    lat_dec = lat_enc
                images_dec, _, _ = decoder(lat_dec)
                images_dec_l.append(images_dec)
            images_dec = torch.cat(images_dec_l)

            model_manager.log_images(images,  'all_input')
            model_manager.log_images(images_dec,  'all_dec')

model_manager.save()
print('Training is complete...')
