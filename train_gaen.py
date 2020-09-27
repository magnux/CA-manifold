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
from src.utils.loss_utils import compute_gan_loss, compute_grad_reg, compute_pl_reg, update_reg_params
from src.utils.model_utils import compute_inception_score, get_grad_norm
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a GAEN')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
d_reg_param = config['training']['d_reg_param']
d_reg_every = config['training']['d_reg_every']
g_reg_every = config['training']['g_reg_every']
alt_reg = config['training']['alt_reg'] if 'alt_reg' in config['training'] else False
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
    'labs_encoder': {'class': 'base', 'sub_class': 'LabsEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'InjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'Discriminator'},
}
model_manager = ModelManager('gaen', networks_dict, config)
encoder = model_manager.get_network('encoder')
labs_encoder = model_manager.get_network('labs_encoder')
decoder = model_manager.get_network('decoder')
generator = model_manager.get_network('generator')
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

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


total_it = config['training']['n_epochs'] * (len(trainloader) // batch_split)

d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', 1 if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)
d_reg_ratio = 1

g_reg_every_enc = model_manager.log_manager.get_last('regs', 'g_reg_every_enc', 1 if g_reg_every > 0 else 0)
g_reg_every_enc_next = g_reg_every_enc
g_reg_param_enc = model_manager.log_manager.get_last('regs', 'g_reg_param_enc', d_reg_param)

g_reg_every_dec = model_manager.log_manager.get_last('regs', 'g_reg_every_dec', 1 if g_reg_every > 0 else 0)
g_reg_every_dec_next = g_reg_every_dec
g_reg_param_dec = model_manager.log_manager.get_last('regs', 'g_reg_param_dec', d_reg_param)

pl_mean_enc = model_manager.log_manager.get_last('regs', 'pl_mean_enc', 0.)
pl_mean_dec = model_manager.log_manager.get_last('regs', 'pl_mean_dec', 0.)

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
                loss_gen_enc_sum, loss_gen_dec_sum = 0, 0

                reg_dis_enc_sum, reg_dis_dec_sum = 0, 0
                reg_gen_enc_sum, reg_gen_dec_sum = 0, 0

                if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                    d_reg_factor = (d_reg_every_mean_next - (it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                else:
                    reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                    reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                if g_reg_every_enc > 0 and it % g_reg_every_enc == 0:
                    g_reg_factor_enc = (g_reg_every_enc_next - (it % g_reg_every_enc_next)) * (1 / g_reg_param_enc)
                else:
                    reg_gen_enc_sum = model_manager.log_manager.get_last('regs', 'reg_gen_enc')

                if g_reg_every_dec > 0 and it % g_reg_every_dec == 0:
                    g_reg_factor_dec = (g_reg_every_dec_next - (it % g_reg_every_dec_next)) * (1 / g_reg_param_dec)
                else:
                    reg_gen_dec_sum = model_manager.log_manager.get_last('regs', 'reg_gen_dec')

                # Discriminator step
                with model_manager.on_step(['dis_encoder', 'discriminator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            lat_labs = labs_encoder(labels)
                            z_enc, _, _ = encoder(images, lat_labs)
                            lat_enc = generator(z_enc, labels)

                        images.requires_grad_()
                        lat_enc.requires_grad_()
                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc, labels)

                        loss_dis_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 1)

                        if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                            reg_dis_enc = (1 / batch_mult) * d_reg_ratio * d_reg_factor * compute_grad_reg(labs_enc, images)
                            model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item() / (d_reg_ratio * d_reg_factor)

                            reg_dis_enc = (1 / batch_mult) * d_reg_ratio * d_reg_factor * compute_grad_reg(labs_enc, lat_enc)
                            model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item() / (d_reg_ratio * d_reg_factor)

                        model_manager.loss_backward(loss_dis_enc, nets_to_train)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            images_dec, _, _ = decoder(lat_gen)

                        lat_gen.requires_grad_()
                        images_dec.requires_grad_()
                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec, labels)

                        loss_dis_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 0)

                        if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                            reg_dis_dec = (1 / batch_mult) * (1 / d_reg_ratio) * d_reg_factor * compute_grad_reg(labs_dec, images_dec)
                            model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            reg_dis_dec_sum += reg_dis_dec.item() / ((1 / d_reg_ratio) * d_reg_factor)

                            reg_dis_dec = (1 / batch_mult) * (1 / d_reg_ratio) * d_reg_factor * compute_grad_reg(labs_dec, lat_gen)
                            model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            reg_dis_dec_sum += reg_dis_dec.item() / ((1 / d_reg_ratio) * d_reg_factor)

                        model_manager.loss_backward(loss_dis_dec, nets_to_train)
                        loss_dis_dec_sum += loss_dis_dec.item()

                    if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                        d_reg_ratio = (reg_dis_enc_sum + 1e-8) / (reg_dis_dec_sum + 1e-8)
                        reg_dis_mean = (reg_dis_enc_sum + reg_dis_dec_sum) / 2
                        loss_dis_mean = (loss_dis_enc_sum + loss_dis_dec_sum) / 2
                        d_reg_every_mean = d_reg_every_mean_next
                        d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every, d_reg_param_mean, d_reg_param,
                                                                                    reg_dis_mean, reg_dis_target, loss_dis_mean)

                    dis_grad_norm = get_grad_norm(discriminator).item()
                    dis_enc_grad_norm = get_grad_norm(dis_encoder).item()

                # Generator step
                with model_manager.on_step(['encoder', 'labs_encoder', 'decoder', 'generator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        lat_labs = labs_encoder(labels)
                        z_enc, out_embs, _ = encoder(images, lat_labs)
                        lat_enc = generator(z_enc, labels)
                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc, labels)

                        if g_reg_every_enc > 0 and it % g_reg_every_enc == 0:
                            if alt_reg:
                                reg_gen_enc = F.mse_loss(lat_enc.norm(), images.norm())
                            else:
                                reg_gen_enc, pl_mean_enc = compute_pl_reg(lat_enc, images, pl_mean_enc)
                            reg_gen_enc = (1 / batch_mult) * g_reg_factor_enc * reg_gen_enc
                            model_manager.loss_backward(reg_gen_enc, nets_to_train, retain_graph=True)
                            reg_gen_enc_sum += reg_gen_enc.item() / g_reg_factor_enc

                        loss_gen_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)
                        model_manager.loss_backward(loss_gen_enc, nets_to_train)
                        loss_gen_enc_sum += loss_gen_enc.item()

                        lat_gen = generator(z_gen, labels)
                        images_dec, _, _ = decoder(lat_gen)
                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec, labels)

                        if not alt_reg and g_reg_every_dec > 0 and it % g_reg_every_dec == 0:
                            reg_gen_dec, pl_mean_dec = compute_pl_reg(images_dec, lat_gen, pl_mean_dec)
                            reg_gen_dec = (1 / batch_mult) * g_reg_factor_dec * reg_gen_dec
                            model_manager.loss_backward(reg_gen_dec, nets_to_train, retain_graph=True)
                            reg_gen_dec_sum += reg_gen_dec.item() / g_reg_factor_dec

                        loss_gen_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                        model_manager.loss_backward(loss_gen_dec, nets_to_train)
                        loss_gen_dec_sum += loss_gen_dec.item()

                    if alt_reg and g_reg_every_enc > 0 and it % g_reg_every_enc == 0:
                        g_reg_every_enc = g_reg_every_enc_next
                        g_reg_every_enc_next, g_reg_param_enc = update_reg_params(g_reg_every_enc_next, g_reg_every,
                                                                                  g_reg_param_enc, d_reg_param,
                                                                                  reg_gen_enc_sum, reg_dis_target)

                    # if alt_reg and g_reg_every_dec > 0 and it % g_reg_every_dec == 0:
                    #     g_reg_every_dec = g_reg_every_dec_next
                    #     g_reg_every_dec_next, g_reg_param_dec = update_reg_params(g_reg_every_dec_next, g_reg_every,
                    #                                                               g_reg_param_dec, d_reg_param,
                    #                                                               reg_gen_dec_sum, reg_dis_target)

                    enc_grad_norm = get_grad_norm(encoder).item()
                    dec_grad_norm = get_grad_norm(decoder).item()
                    gen_grad_norm = get_grad_norm(generator).item()

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    images_gen, _, _ = decoder(lat_gen)

                stream_images(images_gen, config_name + '/gaen', config['training']['out_dir'] + '/gaen')

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_enc_sum + loss_gen_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dis='%.2e' % (np.sum(running_loss_dis) / running_factor),
                              loss_gen='%.2e' % (np.sum(running_loss_gen) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dis_enc', loss_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_dis_dec', loss_dis_dec_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_gen_enc', loss_gen_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_gen_dec', loss_gen_dec_sum, it=it)

                model_manager.log_manager.add_scalar('regs', 'reg_dis_enc', reg_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('regs', 'reg_dis_dec', reg_dis_dec_sum, it=it)
                model_manager.log_manager.add_scalar('regs', 'd_reg_every_mean', d_reg_every_mean, it=it)
                model_manager.log_manager.add_scalar('regs', 'd_reg_param_mean', d_reg_param_mean, it=it)

                if g_reg_every > 0:
                    model_manager.log_manager.add_scalar('regs', 'g_reg_every', g_reg_every, it=it)
                    model_manager.log_manager.add_scalar('regs', 'reg_gen_enc', reg_gen_enc_sum, it=it)
                    model_manager.log_manager.add_scalar('regs', 'reg_gen_dec', reg_gen_dec_sum, it=it)
                    model_manager.log_manager.add_scalar('regs', 'g_reg_every_enc', g_reg_every_enc, it=it)
                    model_manager.log_manager.add_scalar('regs', 'g_reg_param_enc', g_reg_param_enc, it=it)
                    model_manager.log_manager.add_scalar('regs', 'g_reg_every_dec', g_reg_every_dec, it=it)
                    model_manager.log_manager.add_scalar('regs', 'g_reg_param_dec', g_reg_param_dec, it=it)

                    model_manager.log_manager.add_scalar('regs', 'pl_mean_enc', pl_mean_enc, it=it)
                    model_manager.log_manager.add_scalar('regs', 'pl_mean_dec', pl_mean_dec, it=it)

                model_manager.log_manager.add_scalar('norms', 'dis_grad_norm', dis_grad_norm, it=it)
                model_manager.log_manager.add_scalar('norms', 'dis_enc_grad_norm', dis_enc_grad_norm, it=it)
                model_manager.log_manager.add_scalar('norms', 'enc_grad_norm', enc_grad_norm, it=it)
                model_manager.log_manager.add_scalar('norms', 'dec_grad_norm', dec_grad_norm, it=it)
                model_manager.log_manager.add_scalar('norms', 'gen_grad_norm', gen_grad_norm, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, _, trainiter = get_inputs(trainiter, batch_size, device)
            lat_gen = generator(z_test, labels_test)
            images_gen, _, _ = decoder(lat_gen)
            lat_labs = labs_encoder(labels)
            z_enc, _, _ = encoder(images, lat_labs)
            lat_enc = generator(z_enc, labels)
            images_dec, _, _ = decoder(lat_enc)
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
