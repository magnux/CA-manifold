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
from src.utils.model_utils import compute_inception_score, grad_noise
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a GAN')
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
d_reg_param = config['training']['d_reg_param']
d_reg_every = config['training']['d_reg_every']
g_reg_every = config['training']['g_reg_every']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
n_workers = config['training']['n_workers']
z_dim = config['z_dist']['z_dim']
lat_size = lat_size = config['network']['kwargs']['lat_size']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['steps_per_epoch'] = len(trainloader) // batch_split

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], z_dim, device=device)

# Networks
networks_dict = {
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'Discriminator'},
}
# to_avg = ['decoder', 'generator']

model_manager = ModelManager('gan', networks_dict, config)
decoder = model_manager.get_network('decoder')
generator = model_manager.get_network('generator')
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

# decoder_avg = model_manager.get_network_avg('decoder')
# generator_avg = model_manager.get_network_avg('generator')

model_manager.print()

embedding_mat = torch.eye(n_labels, device=device)

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
    if labels.dtype is torch.int64:
        if labels.dim() == 1:
            labels = embedding_mat[labels]
        else:
            labels = labels.to(torch.float32)
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

window_size = math.ceil((len(trainloader) // batch_split) / 10)


d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', d_reg_every if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)

pl_mean_dec = model_manager.log_manager.get_last('regs', 'pl_mean_dec', 0.)

torch.autograd.set_detect_anomaly(True)
g_factor_enc = model_manager.log_manager.get_last('regs', 'g_factor_enc', 0.)
g_factor_dec = model_manager.log_manager.get_last('regs', 'g_factor_dec', 0.)

for epoch in range(model_manager.start_epoch, config['training']['n_epochs']):
    with model_manager.on_epoch(epoch):

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((epoch / config['training']['n_epochs']) * config['training']['batch_mult_steps']) + 1) * batch_split
        # Dynamic reg target for grad annealing
        reg_dis_target = 1. * (1. - 0.999 ** (config['training']['n_epochs'] / (epoch + 1e-8)))
        # Fixed reg target
        # reg_dis_target = 0.1

        it = epoch * (len(trainloader) // batch_split)

        t = trange(len(trainloader) // batch_split)
        t.set_description('| ep: %d | lr: %.2e |' % (epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum = 0, 0
                labs_dis_enc_sign, labs_dis_dec_sign = 0, 0
                loss_gen_dec_sum = 0

                reg_dis_enc_sum, reg_dis_dec_sum = 0, 0
                if g_reg_every > 0:
                    reg_gen_dec_sum = 0
                    pl_mean_dec = 0

                if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                    d_reg_factor = (d_reg_every_mean_next - (it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                else:
                    reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                    reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                # Discriminator step
                with model_manager.on_step(['dis_encoder', 'discriminator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        lat_top_enc, _, _ = dis_encoder(images, labels)
                        labs_enc = discriminator(lat_top_enc, labels)
                        labs_dis_enc_sign += ((1 / batch_mult) * labs_enc.sign().mean()).item()

                        if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                            reg_dis_enc = (1 / batch_mult) * d_reg_factor * compute_grad_reg(labs_enc, images)
                            model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                        loss_dis_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 1)
                        lat_top_enc.register_hook(lambda grad: grad + 10 * (grad - grad.mean(0, keepdims=True)))
                        model_manager.loss_backward(loss_dis_enc, nets_to_train)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            rand_inits = 1e-3 * torch.randn(lat_gen.size(0), n_filter, image_size, image_size, device=device)
                            images_dec, _, _ = decoder(lat_gen, ca_noise=rand_inits)

                        images_dec.requires_grad_()
                        lat_top_dec, _, _ = dis_encoder(images_dec, labels)
                        labs_dec = discriminator(lat_top_dec, labels)
                        labs_dis_dec_sign -= ((1 / batch_mult) * labs_dec.sign().mean()).item()

                        if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                            reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(labs_dec, images_dec)
                            model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                        loss_dis_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 0)
                        lat_top_dec.register_hook(lambda grad: grad + 10 * (grad - grad.mean(0, keepdims=True)))
                        model_manager.loss_backward(loss_dis_dec, nets_to_train)
                        loss_dis_dec_sum += loss_dis_dec.item()

                    if d_reg_every_mean > 0 and it % d_reg_every_mean == 0:
                        reg_dis_mean = 0.5 * (reg_dis_enc_sum + reg_dis_dec_sum)
                        loss_dis_mean = 0.5 * (loss_dis_enc_sum + loss_dis_dec_sum)
                        d_reg_every_mean = d_reg_every_mean_next
                        d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every, d_reg_param_mean,
                                                                                    reg_dis_mean, reg_dis_target, loss_dis_mean)

                    g_factor_enc = np.clip(g_factor_enc + 1e-2 * (labs_dis_enc_sign - 0.2), 0., 1.)
                    g_factor_dec = np.clip(g_factor_dec + 1e-2 * (labs_dis_dec_sign - 0.2), 0., 1.)

                # Generator step
                with model_manager.on_step(['decoder', 'generator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        lat_gen = generator(z_gen, labels)
                        rand_inits = 1e-3 * torch.randn(lat_gen.size(0), n_filter, image_size, image_size, device=device)
                        images_dec, _, _ = decoder(lat_gen, ca_noise=rand_inits)

                        lat_top_dec, _, _ = dis_encoder(images_dec, labels)
                        labs_dec = discriminator(lat_top_dec, labels)

                        if g_reg_every > 0 and it % g_reg_every == 0:
                            reg_gen_dec, pl_mean_dec = compute_pl_reg(images_dec, lat_gen, pl_mean_dec)
                            reg_gen_dec = (1 / batch_mult) * g_reg_every * reg_gen_dec
                            model_manager.loss_backward(reg_gen_dec, nets_to_train, retain_graph=True)
                            reg_gen_dec_sum += reg_gen_dec.item() / g_reg_every

                        loss_gen_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                        lat_top_dec.register_hook(lambda grad: grad + 10 * (grad - grad.mean(0, keepdims=True)))
                        model_manager.loss_backward(loss_gen_dec, nets_to_train)
                        loss_gen_dec_sum += loss_gen_dec.item()

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    rand_inits = 1e-3 * torch.randn(lat_gen.size(0), n_filter, image_size, image_size, device=device)
                    images_gen, _, _ = decoder(lat_gen, ca_noise=rand_inits)

                stream_images(images_gen, config_name + '/gan', config['training']['out_dir'] + '/gan')

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dis='%.2e' % (np.sum(running_loss_dis) / running_factor),
                              loss_gen='%.2e' % (np.sum(running_loss_gen) / running_factor))

                # Log progress
                model_manager.log_manager.add_scalar('learning_rates', 'all', model_manager.lr, it=it)
                if model_manager.momentum is not None:
                    model_manager.log_manager.add_scalar('learning_rates', 'all_mom', model_manager.momentum, it=it)

                model_manager.log_manager.add_scalar('losses', 'loss_dis_enc', loss_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'labs_dis_enc_sign', labs_dis_enc_sign, it=it)
                model_manager.log_manager.add_scalar('losses', 'labs_dis_dec_sign', labs_dis_dec_sign, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_dis_dec', loss_dis_dec_sum, it=it)
                model_manager.log_manager.add_scalar('losses', 'loss_gen_dec', loss_gen_dec_sum, it=it)

                model_manager.log_manager.add_scalar('regs', 'g_factor_enc', g_factor_enc, it=it)
                model_manager.log_manager.add_scalar('regs', 'g_factor_dec', g_factor_dec, it=it)
                model_manager.log_manager.add_scalar('regs', 'reg_dis_enc', reg_dis_enc_sum, it=it)
                model_manager.log_manager.add_scalar('regs', 'reg_dis_dec', reg_dis_dec_sum, it=it)
                model_manager.log_manager.add_scalar('regs', 'd_reg_every_mean', d_reg_every_mean, it=it)
                model_manager.log_manager.add_scalar('regs', 'd_reg_param_mean', d_reg_param_mean, it=it)

                if g_reg_every > 0:
                    model_manager.log_manager.add_scalar('regs', 'reg_gen_dec', reg_gen_dec_sum, it=it)

                    model_manager.log_manager.add_scalar('regs', 'pl_mean_dec', pl_mean_dec, it=it)

                it += 1

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((epoch + 1) % config['training']['sample_every']) == 0:
            t.write('Creating samples...')
            images, labels, _, trainiter = get_inputs(trainiter, batch_size, device)
            lat_gen = generator(z_test, labels_test)
            images_gen, _, _ = decoder(lat_gen)
            model_manager.log_manager.add_imgs(images, 'all_input', it)
            model_manager.log_manager.add_imgs(images_gen, 'all_gen', it)
            for lab in range(config['training']['sample_labels']):
                if labels_test.dim() == 1:
                    fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                else:
                    fixed_lab = labels_test.clone()
                    fixed_lab[:, lab] = 1
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
