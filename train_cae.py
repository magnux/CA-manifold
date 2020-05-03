# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import save_images, rand_erase_images, rand_change_letters, rand_circle_masks
from src.utils.model_utils import ca_seed, SamplePool
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

parser = argparse.ArgumentParser(description='Train a FractalNet')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
letter_encoding = config['network']['kwargs']['letter_encoding']
use_sample_pool = config['training']['sample_pool']
damage_init = config['training']['damage_init']
batch_size = config['training']['batch_size']
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

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
    next_inputs = next(trainiter, None)
    if trainiter is None or next_inputs is None:
        trainiter = iter(trainloader)
        next_inputs = next(trainiter, None)
    images, labels = next_inputs
    images, labels = images[:batch_size, ...], labels[:batch_size, ...]
    images, labels = images.to(device), labels.to(device)
    images.detach().requires_grad_()
    return images, labels, trainiter


images_test, labels_test, trainiter = get_inputs(iter(trainloader), batch_size, device)

if use_sample_pool:
    n_slots = len(trainloader) * 16
    target = []
    for _ in range((n_slots // batch_size) + 1):
        images, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        target.append(images)
    target = torch.cat(target, dim=0)[:n_slots, ...].numpy()
    seed = ca_seed(n_slots, n_filter, image_size, torch.device('cpu')).numpy()
    sample_pool = SamplePool(target=target, init=seed)
    init_seed = ca_seed(batch_size // 16, n_filter, image_size, torch.device('cpu')).numpy()

window_size = len(trainloader) // 10

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss = np.zeros(window_size)

        batch_mult = int((epoch / config['training']['n_epochs']) * 4) + 1

        it = (epoch * len(trainloader))

        t = trange(len(trainloader))
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum = 0

                with model_manager.on_step(['encoder', 'decoder'] + (['letter_encoder', 'letter_decoder'] if letter_encoding else [])):

                    for _ in range(batch_mult):

                        if use_sample_pool:
                            pool_samples = sample_pool.sample(batch_size)
                            images, init_samples = pool_samples.target, pool_samples.init
                            loss_init = np.mean(np.square(images - init_samples[:, :images.shape[1], :, :]), axis=(1,2,3))
                            loss_rank = loss_init.argsort()[::-1]
                            bad_frac = (np.sum(loss_init > 5e-2) // (batch_size // 16))
                            images = images[loss_rank]
                            pool_samples.target[:] = images
                            init_samples = init_samples[loss_rank]
                            for i in range(max(1, bad_frac)):
                                init_samples[(batch_size // 16) * i:(batch_size // 16) * (i + 1), ...] = init_seed
                            images = torch.tensor(images, device=device, requires_grad=True)
                            init_samples = torch.tensor(init_samples, device=device, requires_grad=True)
                            if damage_init:
                                init_samples = rand_circle_masks(init_samples, batch_size // 16)
                        else:
                            images, _, trainiter = get_inputs(trainiter, batch_size, device)
                            init_samples = None

                        # Obscure the input
                        re_images = rand_erase_images(images)

                        # Encoding
                        lat_enc, _, _ = encoder(re_images)

                        if letter_encoding:
                            letters = letter_encoder(lat_enc)
                            letters = rand_change_letters(letters)
                            lat_dec = letter_decoder(letters)
                        else:
                            lat_dec = lat_enc + (1e-3 * torch.randn_like(lat_enc))

                        # Decoding
                        _, out_embs, images_redec_raw = decoder(lat_dec, init_samples)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_redec_raw, images)
                        loss_dec.backward()
                        loss_dec_sum += loss_dec.item()

                        if use_sample_pool:
                            pool_samples.init[:] = out_embs[-1].cpu().detach().numpy()
                            pool_samples.commit()

                # Streaming Images
                with torch.no_grad():
                    lat_enc, _, _ = encoder(images_test)
                    if letter_encoding:
                        letters = letter_encoder(lat_enc)
                        lat_dec = letter_decoder(letters)
                    else:
                        lat_dec = lat_enc
                    images_dec, _, _ = decoder(lat_dec)

                stream_images(images_dec, config_name)

                # Print progress
                running_loss[batch % window_size] = loss_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dec', loss_dec_sum, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, trainiter = get_inputs(trainiter, config['training']['batch_size'], device)
            lat_enc, _, _ = encoder(images)
            if letter_encoding:
                letters = letter_encoder(lat_enc)
                lat_dec = letter_decoder(letters)
            else:
                lat_dec = lat_enc
            images_dec, _, _ = decoder(lat_dec)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)

time.sleep(30)
print('Training is complete...')
