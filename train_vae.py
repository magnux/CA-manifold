# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.distributions import get_zdist
from src.inputs import get_dataset
from src.utils.model_utils import compute_inception_score, grad_ema_update
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext, join

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a VAE')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')


image_size = config['data']['image_size']
channels = config['data']['channels']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
lat_size = config['network']['kwargs']['lat_size']
n_epochs = config['training']['n_epochs']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
batch_mult_steps = config['training']['batch_mult_steps']
n_workers = config['training']['n_workers']
z_dim = config['z_dist']['z_dim']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['steps_per_epoch'] = len(trainloader) // batch_split

# Distributions
zdist = get_zdist(config['z_dist']['type'], z_dim, device=device)


# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'var_encoder': {'class': 'base', 'sub_class': 'VarEncoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
model_manager = ModelManager('vae', networks_dict, config)
encoder = model_manager.get_network('encoder')
var_encoder = model_manager.get_network('var_encoder')
generator = model_manager.get_network('generator')
decoder = model_manager.get_network('decoder')

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
    z_gen = zdist.sample((images.size(0),))
    z_gen.detach_().requires_grad_()
    return images, labels, z_gen, trainiter


images_test, labels_test, _, trainiter = get_inputs(iter(trainloader), batch_size, device)


if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()


window_size = math.ceil((len(trainloader) // batch_split) / 10)

for epoch in range(model_manager.start_epoch, n_epochs):
    with model_manager.on_epoch(epoch):

        running_loss_dec = np.zeros(window_size)

        batch_mult = (int((epoch / n_epochs) * batch_mult_steps) + 1) * batch_split

        it = (epoch * (len(trainloader) // batch_split))

        t = trange(len(trainloader) // batch_split)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum, loss_kl_sum = 0, 0

                with model_manager.on_step(['encoder', 'var_encoder', 'generator', 'decoder']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, _, trainiter = get_inputs(trainiter, batch_split_size, device)

                        lat_enc, _, _ = encoder(images, labels)
                        z_enc, loss_kl = var_encoder(lat_enc)

                        model_manager.loss_backward((1 / batch_mult) * loss_kl.mean(), nets_to_train, retain_graph=True)
                        loss_kl_sum += loss_kl.mean().item()

                        lat_dec = generator(z_enc, labels)
                        images_dec, _, _ = decoder(lat_dec)

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_dec, images)
                        model_manager.loss_backward(loss_dec, nets_to_train)
                        loss_dec_sum += loss_dec.item()

                # if isinstance(generator, torch.nn.DataParallel):
                #     grad_ema_update(var_encoder.module)
                # else:
                #     grad_ema_update(var_encoder)

                # Streaming Images
                with torch.no_grad():
                    lat_enc, _, _ = encoder(images_test, labels_test)
                    z_enc, _ = var_encoder(lat_enc)
                    lat_dec = generator(z_enc, labels_test)
                    images_dec, _, _ = decoder(lat_dec)

                stream_images(images_dec, config_name + '/vae', config['training']['out_dir'] + '/vae')

                # Print progress
                running_loss_dec[batch % window_size] = loss_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dec='%.2e' % (np.sum(running_loss_dec) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)
                if model_manager.momentum is not None:
                    model_manager.log_manager.add_scalar('learning_rates', 'all_mom', model_manager.momentum, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dec', loss_dec_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_kl', loss_kl_sum, it=it)

                it += 1

    # save_multigauss_params(zdist_mu, zdist_cov, join(config['training']['out_dir'], 'vae'))

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, z_gen, trainiter = get_inputs(trainiter, batch_size, device)
            # images_gen, _, _ = decoder(lat_gen)
            lat_enc, _, _ = encoder(images, labels)
            z_enc, _ = var_encoder(lat_enc)
            lat_dec = generator(z_enc, labels)
            images_dec, _, _ = decoder(lat_dec)
            lat_gen = generator(z_gen, labels)
            images_gen, _, _ = decoder(lat_gen)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)
            model_manager.log_manager.add_imgs(images_gen, 'all_gen', it)
            for lab in range(config['training']['sample_labels']):
                if labels.dim() == 1:
                    fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                else:
                    fixed_lab = labels.clone()
                    fixed_lab[:, lab] = 1
                lat_gen = generator(z_gen, fixed_lab)
                images_gen, _, _ = decoder(lat_gen)
                model_manager.log_manager.add_imgs(images_gen, 'class_%04d' % lab, it)

        # Perform inception
        if config['training']['inception_every'] > 0 and ((epoch + 1) % config['training']['inception_every']) == 0 and epoch > 0:
            t.write('Computing inception/fid!')
            inception_mean, inception_std, fid = compute_inception_score(None, decoder,
                                                                         10000, 10000, config['training']['batch_size'],
                                                                         zdist, None, fid_real_samples, device)
            model_manager.log_manager.add_scalar('inception_score', 'mean', inception_mean, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'stddev', inception_std, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'fid', fid, it=it)

print('Training is complete...')
