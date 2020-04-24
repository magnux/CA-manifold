# -*- coding: utf-8 -*-
from os import path
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.model_utils import get_nsamples
from src.utils.media_utils import save_images, ran_erase_images, noise_letters
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images

parser = argparse.ArgumentParser(description='Train a FractalNet')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=config['training']['batch_size'],
                                          shuffle=True, num_workers=config['training']['nworkers'], drop_last=True)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoder'},
    'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoder'},
}
model_manager = ModelManager('cae', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
letter_encoder = model_manager.get_network('letter_encoder')
letter_decoder = model_manager.get_network('letter_decoder')

model_manager.print()


def get_inputs(trainiter, batch_size):
    next_inputs = next(trainiter, None)
    if trainiter is None or next_inputs is None:
        trainiter = iter(trainloader)
        next_inputs = next(trainiter, None)
    images, labels = next_inputs
    images, labels = images[:batch_size, ...], labels[:batch_size, ...]
    images, labels = images.to(device), labels.to(device)
    images.detach_().requires_grad_()
    return images, labels, trainiter


window_size = len(trainloader) // 10
batch_size = config['training']['batch_size']
images_test, labels_test, trainiter = get_inputs(iter(trainloader), batch_size)

for epoch in range(model_manager.start_epoch, config['training']['nepochs']):
    with model_manager.on_epoch(epoch):

        running_loss = np.zeros(window_size)

        batch_mult = int((epoch / config['training']['nepochs']) * 4) + 1

        it = (epoch * len(trainloader))

        t = trange(len(trainloader))
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum = 0
                
                with model_manager.on_step(['encoder', 'decoder', 'letter_encoder', 'letter_decoder']):

                    for _ in range(batch_mult):

                        images, labels, trainiter = get_inputs(trainiter, batch_size)

                        # Adding to the input
                        re_images = ran_erase_images(images)

                        # Encoding
                        lat_enc, _, _ = encoder(re_images)
                        letters = letter_encoder(lat_enc)

                        # Adding to the latent space (encoded in letters)
                        letters = noise_letters(letters)

                        # Decoding
                        lat_dec = letter_decoder(letters)
                        _, _, images_redec_raw = decoder(lat_dec)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_redec_raw, images)
                        loss_dec.backward()
                        loss_dec_sum += loss_dec.item()

                # Streaming Images
                with torch.no_grad():
                    lat_enc, _, _ = encoder(images_test)
                    letters = letter_encoder(lat_enc)
                    lat_dec = letter_decoder(letters)
                    images_dec, _, _ = decoder(lat_dec)

                stream_images(images_dec)

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
            images, labels, trainiter = get_inputs(trainiter, config['training']['batch_size'])
            lat_enc, _, _ = encoder(images)
            letters = letter_encoder(lat_enc)
            lat_dec = letter_decoder(letters)
            images_dec, _, _ = decoder(lat_dec)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)

print('Training is complete...')
