# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.distributions import get_ydist, get_zdist
from src.inputs import get_dataset
from src.utils.loss_utils import discretized_mix_logistic_loss, sample_gaussian, gaussian_kl_loss, compute_gan_loss
from src.utils.model_utils import compute_inception_score, ca_seed
from src.utils.media_utils import rand_erase_images, rand_change_letters
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a IRMAE')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

config['network']['kwargs']['irm'] = True

image_size = config['data']['image_size']
channels = config['data']['channels']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
n_workers = config['training']['n_workers']
z_dim = config['z_dist']['z_dim']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], (4, 16 * config['network']['kwargs']['lat_size']), device=device)


# Networks
networks_dict = {
    'labs_encoder': {'class': 'base', 'sub_class': 'LabsEncoder'},
    'encoder': {'class': config['network']['class'], 'sub_class': 'InjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'irm_generator': {'class': 'base', 'sub_class': 'IRMGenerator'},
    'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoder'},
    'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoder'},
}
model_manager = ModelManager('irmae', networks_dict, config)
labs_encoder = model_manager.get_network('labs_encoder')
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
irm_generator = model_manager.get_network('irm_generator')
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
    images, labels = images.to(device), labels.to(device)
    images = images.detach().requires_grad_()
    z_gen = torch.softmax(zdist.sample((images.size(0),)) * 10., dim=1)
    z_gen.detach_().requires_grad_()
    return images, labels, z_gen, trainiter


images_test, labels_test, z_test, trainiter = get_inputs(iter(trainloader), batch_size, device)


if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()


def generator(z, labels):
    lat_gen_let = letter_decoder(z)
    lat_gen = irm_generator(lat_gen_let, labels)
    return lat_gen

window_size = math.ceil((len(trainloader) // batch_split) / 10)

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss_cent = np.zeros(window_size)
        running_loss_dec = np.zeros(window_size)

        batch_mult = (int((epoch / config['training']['n_epochs']) * config['training']['batch_mult_steps']) + 1) * batch_split

        it = (epoch * (len(trainloader) // batch_split))

        t = trange(len(trainloader) // batch_split)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_cent_sum, loss_dec_sum = 0, 0

                with model_manager.on_step(['labs_encoder', 'encoder', 'decoder', 'irm_generator', 'letter_encoder', 'letter_decoder']):

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        # re_images = rand_erase_images(images)
                        lat_labs = labs_encoder(labels)
                        lat_enc, out_embs, _ = encoder(images, lat_labs)

                        lat_enc_let = letter_encoder(lat_enc)
                        lat_enc_let = rand_change_letters(lat_enc_let)
                        lat_dec = letter_decoder(lat_enc_let)
                        lat_dec = irm_generator(lat_dec, labels)

                        images_dec, _, images_dec_raw = decoder(lat_dec)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_dec_raw, images)
                        loss_dec.backward(retain_graph=True)
                        loss_dec_sum += loss_dec.item()

                        loss_cent = loss_cent.mean()
                        loss_cent = (1 / batch_mult) * 1e-1 * loss_cent
                        loss_cent.backward()
                        loss_cent_sum += loss_cent.item()

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    images_gen, _, _ = decoder(lat_gen)

                stream_images(images_gen, config_name + '/irmae', config['training']['out_dir'] + '/irmae')

                # Print progress
                running_loss_cent[batch % window_size] = loss_cent_sum
                running_loss_dec[batch % window_size] = loss_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_cent='%.2e' % (np.sum(running_loss_cent) / running_factor),
                              loss_dec='%.2e' % (np.sum(running_loss_dec) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_cent', loss_cent_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_dec', loss_dec_sum, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, z_gen, trainiter = get_inputs(trainiter, batch_size, device)
            lat_gen = generator(z_test, labels_test)
            images_gen, _, _ = decoder(lat_gen)
            lat_labs = labs_encoder(labels)
            lat_enc, out_embs, _ = encoder(images)
            lat_enc_let = letter_encoder(lat_enc)
            lat_dec, _ = letter_decoder(lat_enc_let, labels)
            lat_dec = irm_generator(lat_dec, labels)
            images_dec, _, _ = decoder(lat_dec)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_gen, 'all_gen', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)
            for lab in range(config['training']['sample_labels']):
                fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                lat_gen = generator(z_test, fixed_lab)
                images_gen, _, _ = decoder(lat_gen)
                model_manager.log_manager.add_imgs(images_gen, 'class_%04d' % lab, it)

        # Perform inception
        if config['training']['inception_every'] > 0 and ((epoch + 1) % config['training']['inception_every']) == 0 and epoch > 0:
            t.write('Computing inception/fid!')
            inception_mean, inception_std, fid = compute_inception_score(generator, decoder,
                                                                         10000, 10000, config['training']['batch_size'],
                                                                         zdist, ydist, fid_real_samples, device)
            model_manager.log_manager.add_scalar('inception_score', 'mean', inception_mean, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'stddev', inception_std, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'fid', fid, it=it)

print('Training is complete...')
