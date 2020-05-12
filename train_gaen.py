# -*- coding: utf-8 -*-
from os import path
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.distributions import get_ydist, get_zdist
from src.inputs import get_dataset
from src.utils.loss_utils import compute_gan_loss, compute_grad2
from src.utils.model_utils import compute_inception_score, toggle_grad, zero_grad, bkp_grad, copy_grad_bkp, del_grad_bkp, apply_grad_bkp
from src.utils.media_utils import save_images
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
n_calls = config['network']['kwargs']['n_calls']
reg_param = config['training']['reg_param']
batch_size = config['training']['batch_size']
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], config['z_dist']['z_dim'], device=device)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'InjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'Discriminator'},
}
model_manager = ModelManager('gaen', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
generator = model_manager.get_network('generator')
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

model_manager.print()


def get_inputs(trainiter, batch_size, device):
    next_inputs = next(trainiter, None)
    if trainiter is None or next_inputs is None:
        trainiter = iter(trainloader)
        next_inputs = next(trainiter, None)
    images, labels = next_inputs
    images, labels = images[:batch_size, ...], labels[:batch_size, ...]
    images, labels = images.to(device), labels.to(device)
    images = images.detach().requires_grad_()
    z_gen = zdist.sample((images.size(0),)).clamp_(-3, 3)
    z_gen.detach_().requires_grad_()
    return images, labels, z_gen, trainiter


images_test, labels_test, z_test, trainiter = get_inputs(iter(trainloader), batch_size, device)

if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()


window_size = len(trainloader) // 10

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = int((epoch / config['training']['n_epochs']) * 4) + 1

        it = (epoch * len(trainloader))

        t = trange(len(trainloader))
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum, reg_dis_enc_sum, reg_dis_dec_sum = 0, 0, 0, 0
                loss_gen_enc_sum, loss_gen_dec_sum = 0, 0

                # Discriminator step
                with model_manager.on_step(['dis_encoder', 'discriminator']):

                    for _ in range(batch_mult):

                        images, labels, _, trainiter = get_inputs(trainiter, batch_size, device)

                        with torch.no_grad():
                            lat_enc, _, _ = encoder(images)

                        lat_enc.requires_grad_()
                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc, labels)

                        loss_dis_enc = (1/batch_mult) * compute_gan_loss(labs_enc, 1)

                        reg_dis_enc = (1/batch_mult) * reg_param * compute_grad2(labs_enc, images).mean()
                        reg_dis_enc.backward(retain_graph=True)
                        reg_dis_enc_sum += reg_dis_enc.item()

                        reg_dis_enc = (1 / batch_mult) * reg_param * compute_grad2(labs_enc, lat_enc).mean()
                        reg_dis_enc.backward(retain_graph=True)
                        reg_dis_enc_sum += reg_dis_enc.item()

                        loss_dis_enc.backward()
                        loss_dis_enc_sum += loss_dis_enc.item()

                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_size, device)

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            images_dec, _, _ = decoder(lat_gen)

                        images_dec.requires_grad_()
                        lat_gen.requires_grad_()
                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec, labels)

                        loss_dis_dec = (1/batch_mult) * compute_gan_loss(labs_dec, 0)

                        reg_dis_dec = (1 / batch_mult) * reg_param * compute_grad2(labs_dec, images_dec).mean()
                        reg_dis_dec.backward(retain_graph=True)
                        reg_dis_dec_sum += reg_dis_dec.item()

                        reg_dis_dec = (1 / batch_mult) * reg_param * compute_grad2(labs_dec, lat_gen).mean()
                        reg_dis_dec.backward(retain_graph=True)
                        reg_dis_dec_sum += reg_dis_dec.item()
                        
                        loss_dis_dec.backward()
                        loss_dis_dec_sum += loss_dis_dec.item()

                # Generator step
                with model_manager.on_step(['encoder', 'decoder', 'generator']):

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_size, device)

                        lat_enc, _, _ = encoder(images)
                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc, labels)

                        loss_gen_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)
                        loss_gen_enc.backward()
                        loss_gen_enc_sum += loss_gen_enc.item()

                        # Gradient alignment
                        bkp_grad(encoder, 'loss_gan')
                        zero_grad(encoder)

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            images_gen, _, _ = decoder(lat_gen)

                        lat_reenc, _, _ = encoder(images_gen)
                        loss_reenc = (1 / batch_mult) * F.mse_loss(lat_reenc, lat_gen)
                        loss_reenc.backward()
                        bkp_grad(encoder, 'loss_l2')
                        zero_grad(encoder)
                        apply_grad_bkp(encoder, 'loss_l2', torch.abs)
                        apply_grad_bkp(encoder, 'loss_l2', lambda x: x / (1.0 + x.norm()))
                        apply_grad_bkp(encoder, 'loss_l2', lambda x: x + 1.0)
                        apply_grad_bkp(encoder, 'loss_l2', 'loss_gan', lambda x, y: x * y)
                        copy_grad_bkp(encoder, 'loss_l2')
                        del_grad_bkp(encoder)

                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_size, device)

                        lat_gen = generator(z_gen, labels)
                        images_dec, _, images_dec_raw = decoder(lat_gen)
                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec, labels)

                        loss_gen_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                        loss_gen_dec.backward()
                        loss_gen_dec_sum += loss_gen_dec.item()

                        # Gradient alignment
                        bkp_grad(decoder, 'loss_gan')
                        zero_grad(decoder)

                        with torch.no_grad():
                            lat_enc, _, _ = encoder(images)

                        images_dec, _, _ = decoder(lat_enc)
                        loss_redec = (1 / batch_mult) * F.mse_loss(images_dec, images)
                        loss_redec.backward()
                        bkp_grad(decoder, 'loss_l2')
                        zero_grad(decoder)
                        apply_grad_bkp(decoder, 'loss_l2', torch.abs)
                        apply_grad_bkp(decoder, 'loss_l2', lambda x: x / (1.0 + x.norm()))
                        apply_grad_bkp(decoder, 'loss_l2', lambda x: x + 1.0)
                        apply_grad_bkp(decoder, 'loss_l2', 'loss_gan', lambda x, y: x * y)
                        copy_grad_bkp(decoder, 'loss_l2')
                        del_grad_bkp(decoder)

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    images_gen, _, _ = decoder(lat_gen)

                stream_images(images_gen, config_name, config['training']['out_dir'])

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

                model_manager.log_manager.add_scalar('losses', 'reg_dis_enc', reg_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'reg_dis_dec', reg_dis_dec_sum, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_gen_enc', loss_gen_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_gen_dec', loss_gen_dec_sum, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, _, trainiter = get_inputs(trainiter, config['training']['batch_size'], device)
            lat_gen = generator(z_test, labels_test)
            images_gen, _, _ = decoder(lat_gen)
            lat_enc, _, _ = encoder(images)
            images_dec, _, _ = decoder(lat_enc)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_gen, 'all_gen', it)
            model_manager.log_manager.add_imgs(images_dec, 'all_dec', it)
            for lab in range(config['training']['sample_labels']):
                fixed_lab = torch.full((config['training']['batch_size'],), lab, device=device, dtype=torch.int64)
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
