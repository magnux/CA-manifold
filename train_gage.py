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
from src.utils.loss_utils import age_gaussian_kl_loss
from src.utils.model_utils import compute_inception_score, get_grad_norm
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a gage')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['z_dim'], device=device)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'ZInjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
}
model_manager = ModelManager('gage', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
generator = model_manager.get_network('generator')

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


images_test, labels_test, z_test, trainiter = get_inputs(iter(trainloader), batch_size, device)


if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()


total_it = config['training']['n_epochs'] * (len(trainloader) // batch_split)

window_size = math.ceil((len(trainloader) // batch_split) / 10)

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((epoch / config['training']['n_epochs']) * config['training']['batch_mult_steps']) + 1) * batch_split
        reg_dis_target = 1e-3 * ((1 + 1e-3) - (epoch / config['training']['n_epochs']))
        it = epoch * (len(trainloader) // batch_split)

        t = trange(len(trainloader) // batch_split)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum = 0, 0
                loss_gen_dec_sum = 0
                loss_enc_sum, loss_dec_sum = 0, 0

                # Discriminator step
                with model_manager.on_step(['encoder']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        z_enc, _, _ = encoder(images, labels)
                        loss_dis_enc = (1 / batch_mult) * age_gaussian_kl_loss(z_enc)

                        model_manager.loss_backward(loss_dis_enc, nets_to_train)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            images_dec, out_embs, _ = decoder(lat_gen)
                            images_redec, _, _ = decoder(lat_gen, out_embs[-1])

                        z_dec, _, _ = encoder(images_dec, labels)
                        loss_dis_dec = (1 / batch_mult) * -age_gaussian_kl_loss(z_dec)

                        model_manager.loss_backward(loss_dis_dec, nets_to_train)
                        loss_dis_dec_sum += loss_dis_dec.item()

                        z_redec, _, _ = encoder(images_redec, labels)

                        loss_enc = (1 / batch_mult) * F.mse_loss(z_redec, z_gen)
                        model_manager.loss_backward(loss_enc, nets_to_train)
                        loss_enc_sum += loss_enc.item()

                # Decoder step
                with model_manager.on_step(['decoder', 'generator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, _, trainiter = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            z_enc, _, _ = encoder(images, labels)

                        lat_enc = generator(z_enc, labels)
                        images_dec, out_embs, _ = decoder(lat_enc)
                        z_dec, _, _ = encoder(images, labels)

                        loss_gen_dec = (1 / batch_mult) * age_gaussian_kl_loss(z_dec)
                        model_manager.loss_backward(loss_gen_dec, nets_to_train, retain_graph=True)
                        loss_gen_dec_sum += loss_gen_dec.item()

                        images_redec, _, _ = decoder(lat_enc, out_embs[-1])

                        loss_dec = (1 / batch_mult) * F.mse_loss(images_redec, images)
                        model_manager.loss_backward(loss_dec, nets_to_train)
                        loss_dec_sum += loss_dec.item()

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    images_gen, out_embs, _ = decoder(lat_gen)
                    images_regen, _, _ = decoder(lat_gen, out_embs[-1])
                    images_gen = torch.cat([images_gen, images_regen], dim=3)

                stream_images(images_gen, config_name + '/gage', config['training']['out_dir'] + '/gage')

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dis='%.2e' % (np.sum(running_loss_dis) / running_factor),
                              loss_gen='%.2e' % (np.sum(running_loss_gen) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dis_enc', loss_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_dis_dec', loss_dis_dec_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_gen_dec', loss_gen_dec_sum, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_enc', loss_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_dec', loss_dec_sum, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, _, trainiter = get_inputs(trainiter, batch_size, device)
            lat_gen = generator(z_test, labels_test)
            images_gen, out_embs, _ = decoder(lat_gen)
            images_regen, _, _ = decoder(lat_gen, out_embs[-1])
            images_gen = torch.cat([images_gen, images_regen], dim=3)
            z_enc, _, _ = encoder(images, labels)
            lat_enc = generator(z_enc, labels)
            images_dec, out_embs, _ = decoder(lat_enc)
            images_redec, _, _ = decoder(lat_enc, out_embs[-1])
            images_dec = torch.cat([images_dec, images_redec], dim=3)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_gen, 'all_gen', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)
            for lab in range(config['training']['sample_labels']):
                if labels_test.dim() == 1:
                    fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                else:
                    fixed_lab = labels_test.clone()
                    fixed_lab[:, lab] = 1
                lat_gen = generator(z_test, fixed_lab)
                images_gen, out_embs, _ = decoder(lat_gen)
                images_regen, _, _ = decoder(lat_gen, out_embs[-1])
                images_gen = torch.cat([images_gen, images_regen], dim=3)
                model_manager.log_manager.add_imgs(images_gen, 'class_%04d' % lab, it)

        # Perform inception
        if config['training']['inception_every'] > 0 and ((epoch + 1) % config['training']['inception_every']) == 0 and epoch > 0:
            t.write('Computing inception/fid!')
            inception_mean, inception_std, fid = compute_inception_score(generator, decoder,
                                                                         10000, 10000, config['training']['batch_size'],
                                                                         zdist, ydist, fid_real_samples, device, 2)
            model_manager.log_manager.add_scalar('inception_score', 'mean', inception_mean, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'stddev', inception_std, it=it)
            model_manager.log_manager.add_scalar('inception_score', 'fid', fid, it=it)

print('Training is complete...')
