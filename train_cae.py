# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import rand_erase_images, rand_change_letters, rand_circle_masks
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
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
n_filter = config['network']['kwargs']['n_filter']
letter_encoding = config['network']['kwargs']['letter_encoding']
persistence = config['training']['persistence']
regeneration = config['training']['regeneration']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['steps_per_epoch'] = len(trainloader) // batch_split

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'Encoder'},
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


def get_inputs(trainiter, batch_size, device):
    images, labels = [], []
    n_batches = math.ceil(batch_size / batch_split_size)
    for _ in range(n_batches):
        next_inputs = next(trainiter, None)
        if trainiter is None or next_inputs is None:
            trainiter = iter(trainloader)
            next_inputs = next(trainiter, None)
        images.append(next_inputs[0])
        labels.append(next_inputs[1])
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    if batch_size % config['training']['batch_size'] > 0:
        images, labels = images[:batch_size, ...], labels[:batch_size, ...]
    images = (images + 1. / 128 * torch.randn_like(images)).clamp_(-1.0, 1.0)
    images, labels = images.to(device), labels.to(device)
    return images, labels, trainiter


images_test, labels_test, trainiter = get_inputs(iter(trainloader), batch_size, device)

window_size = math.ceil((len(trainloader) // batch_split) / 10)

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss = np.zeros(window_size)

        batch_mult = (int((epoch / config['training']['n_epochs']) * config['training']['batch_mult_steps']) + 1) * batch_split

        it = (epoch * (len(trainloader) // batch_split))

        t = trange(len(trainloader) // batch_split)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum, loss_pers_sum, loss_regen_sum = 0, 0, 0

                with model_manager.on_step(['encoder', 'decoder'] + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])) as nets_to_train:

                    for _ in range(batch_mult):

                        images, _, trainiter = get_inputs(trainiter, batch_split_size, device)
                        init_samples = None

                        # Obscure the input
                        # re_images = rand_erase_images(images)

                        # Encoding
                        lat_enc, _, _ = encoder(images)

                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            letters = rand_change_letters(letters)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc + (1e-3 * torch.randn_like(lat_enc))

                        # Decoding
                        _, out_embs, images_redec_raw = decoder(lat_dec)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_redec_raw, images)
                        model_manager.loss_backward(loss_dec, nets_to_train, retain_graph=True if (persistence or regeneration) else False)
                        loss_dec_sum += loss_dec.item()

                        if persistence:
                            n_calls_save = decoder.n_calls

                            pers_steps = 4
                            decoder.n_calls = pers_steps
                            _, pers_out_embs, _ = decoder(lat_dec, out_embs[-1])

                            pers_target_out_embs = [(out_embs[-1] if o % 2 == 0 else pers_out_embs[1]) for o in range(pers_steps)]

                            loss_pers = (1 / batch_mult) * 100 * F.mse_loss(torch.stack(pers_out_embs[1:]), torch.stack(pers_target_out_embs))
                            model_manager.loss_backward(loss_pers, nets_to_train, retain_graph=True)
                            loss_pers_sum += loss_pers.item()

                            decoder.n_calls = n_calls_save

                        if regeneration:
                            n_calls_save = decoder.n_calls

                            regen_init = np.random.randint(8, 16)
                            regen_steps = 4
                            decoder.n_calls = regen_steps
                            _, regen_out_embs, _ = decoder(lat_dec, rand_circle_masks(out_embs[regen_init - 1], batch_split_size))

                            regen_target_out_embs = out_embs[regen_init:regen_init + regen_steps]

                            loss_regen = (1 / batch_mult) * 100 * F.mse_loss(torch.stack(regen_out_embs[1:]), torch.stack(regen_target_out_embs))
                            model_manager.loss_backward(loss_regen, nets_to_train, retain_graph=True)
                            loss_regen_sum += loss_regen.item()

                            decoder.n_calls = n_calls_save

                # Streaming Images
                with torch.no_grad():
                    lat_enc, _, _ = encoder(images_test)
                    if letter_encoding:
                        letters = letter_encoder(lat_enc)
                        lat_dec = letter_decoder(letters)
                    else:
                        lat_dec = lat_enc
                    images_dec, _, _ = decoder(lat_dec)

                stream_images(images_dec, config_name, config['training']['out_dir'])

                # Print progress
                running_loss[batch % window_size] = loss_dec_sum + loss_pers_sum + loss_regen_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)
                if model_manager.momentum is not None:
                    model_manager.log_manager.add_scalar('learning_rates', 'all_mom', model_manager.momentum, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dec', loss_dec_sum, it=it)
                if persistence:
                    model_manager.log_manager.add_scalar('losses', 'loss_pers', loss_pers_sum, it=it)
                if regeneration:
                    model_manager.log_manager.add_scalar('losses', 'loss_regen', loss_regen_sum, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, trainiter = get_inputs(trainiter, batch_size, device)
            lat_enc, _, _ = encoder(images)
            if letter_encoding:
                letters = letter_encoder(lat_enc)
                lat_dec = letter_decoder(letters)
            else:
                lat_dec = lat_enc
            images_dec, _, _ = decoder(lat_dec)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)

print('Training is complete...')
