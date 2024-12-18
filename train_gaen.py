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
from src.utils.media_utils import rand_circle_masks
from src.utils.loss_utils import compute_grad_reg, compute_gan_loss, update_reg_params, compute_pl_reg, update_g_factors
from src.utils.model_utils import compute_inception_score, grad_mult, grad_mult_hook, grad_dither, grad_dither_hook, update_network_average, get_grads_stats, get_tensor_stats, grad_ema_update
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
config['network']['kwargs']['irm_z'] = True
image_size = config['data']['image_size']
channels = config['data']['channels']
n_labels = config['data']['n_labels']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
n_epochs = config['training']['n_epochs']
d_reg_param = config['training']['d_reg_param']
d_reg_every = config['training']['d_reg_every']
g_reg_every = config['training']['g_reg_every']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
batch_mult_steps = config['training']['batch_mult_steps']
n_workers = config['training']['n_workers']
pre_train = config['training']['pre_train'] if 'pre_train' in config['training'] else False
z_dim = config['z_dist']['z_dim']
lat_size = lat_size = config['network']['kwargs']['lat_size']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], z_dim, device=device)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'ZInjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    # 'redecoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'InjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'UnconditionalDiscriminator'},
}
# to_avg = ['encoder', 'decoder', 'generator']

model_manager = ModelManager('gaen', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
# redecoder = model_manager.get_network('redecoder')
generator = model_manager.get_network('generator')
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

# encoder_avg = model_manager.get_network_avg('encoder')
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

if pre_train:
    for epoch in range(model_manager.epoch, n_epochs // 8):
        with model_manager.on_epoch():
            running_loss_dec = np.zeros(window_size)

            t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
            t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
            for batch in t:
                with model_manager.on_batch():

                    loss_dec_sum = 0

                    with model_manager.on_step(['encoder', 'decoder', 'generator']) as nets_to_train:
                        for _ in range(batch_split):
                            images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                            z_enc, _, _ = encoder(images, labels)
                            lat_enc = generator(z_enc, labels)
                            images_dec, _, _ = decoder(lat_enc)

                            loss_dec = (1 / batch_split) * F.mse_loss(images_dec, images)
                            model_manager.loss_backward(loss_dec, nets_to_train)
                            loss_dec_sum += loss_dec.item()

                # Streaming Images
                with torch.no_grad():
                    z_enc, _, _ = encoder(images_test, labels_test)
                    lat_enc = generator(z_enc, labels_test)
                    images_dec, _, _ = decoder(lat_enc)

                stream_images(images_dec, config_name + '/gaen_pretrain', config['training']['out_dir'] + '/gaen_pretrain')

                # Print progress
                running_loss_dec[batch % window_size] = loss_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dec='%.2e' % (np.sum(running_loss_dec) / running_factor))

                # Log progress
                model_manager.log_scalar('losses',  'loss_dec',  loss_dec_sum)

    print('Pre-training is complete...')
    model_manager.epoch = max(model_manager.epoch, n_epochs // 8)


d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', d_reg_every if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)

pl_mean_enc = model_manager.log_manager.get_last('regs', 'pl_mean_enc', 0.)
pl_mean_dec = model_manager.log_manager.get_last('regs', 'pl_mean_dec', 0.)

torch.autograd.set_detect_anomaly(True)
# g_factor_enc = model_manager.log_manager.get_last('regs', 'g_factor_enc', 1e-3)
# g_factor_dec = model_manager.log_manager.get_last('regs', 'g_factor_dec', 1e-3)

retrain = False
if model_manager.epoch >= n_epochs:
    print('Network is already fully trained, continued onto the retrain phase')
    retrain = True
    n_epochs *= 2

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split
        # Discriminator reg target
        if retrain:
            reg_dis_target = config['training']['lr'] * (1. - 0.999 ** ((n_epochs // 2) / ((epoch - (n_epochs // 2)) + 1e-8)))
        else:
            reg_dis_target = config['training']['lr']
        # Discriminator mean sign target
        sign_mean_target = 0.2  # 0.5 * (1. - 0.9 ** (n_epochs / (epoch + 1e-8)))

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum = 0, 0
                labs_dis_enc_sign, labs_dis_dec_sign = 0, 0
                # loss_dis_unb_sum = 0
                loss_gen_enc_sum, loss_gen_dec_sum = 0, 0
                loss_dec_sum = 0

                reg_dis_enc_sum, reg_dis_dec_sum = 0, 0
                reg_gen_enc_sum, reg_gen_dec_sum = 0, 0

                if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                    d_reg_factor = (d_reg_every_mean_next - (model_manager.it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                else:
                    reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                    reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                if not (g_reg_every > 0 and model_manager.it % g_reg_every == 1):
                    reg_gen_enc_sum = model_manager.log_manager.get_last('regs', 'reg_gen_enc')
                    reg_gen_dec_sum = model_manager.log_manager.get_last('regs', 'reg_gen_dec')

                with model_manager.on_step(['dis_encoder', 'discriminator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        with torch.no_grad():
                            z_enc, _, _ = encoder(images, labels)
                            lat_enc = generator(z_enc, labels)
                            # lat_enc = (1. - g_factor_enc) * lat_enc + g_factor_enc * torch.randn_like(lat_enc)

                        lat_enc.requires_grad_()
                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc)
                        labs_dis_enc_sign += ((1 / batch_mult) * labs_enc.sign().mean()).item()

                        if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                            reg_dis_enc = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_enc, images)
                            model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                            reg_dis_enc = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_enc, lat_enc)
                            model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                        loss_dis_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 1)
                        # labs_enc.register_hook(grad_mult_hook(g_factor_enc))
                        model_manager.loss_backward(loss_dis_enc, nets_to_train)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        with torch.no_grad():
                            lat_gen = generator(z_gen, labels)
                            # lat_gen = (1. - g_factor_dec) * lat_gen + g_factor_dec * torch.randn_like(lat_gen)
                            images_dec, _, _ = decoder(lat_gen)

                        lat_gen.requires_grad_()
                        images_dec.requires_grad_()
                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec)
                        labs_dis_dec_sign -= ((1 / batch_mult) * labs_dec.sign().mean()).item()

                        if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                            reg_dis_dec = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_dec, images_dec)
                            model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                            reg_dis_dec = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_dec, lat_gen)
                            model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                        loss_dis_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 0)
                        # labs_dec.register_hook(grad_mult_hook(g_factor_dec))
                        model_manager.loss_backward(loss_dis_dec, nets_to_train)
                        loss_dis_dec_sum += loss_dis_dec.item()

                        # images_mix = images.clone()
                        # images_mix[batch_split_size//2:, ...].data.copy_(images_dec[batch_split_size//2:, ...])
                        # lat_top_dec, _, _ = dis_encoder(images_mix, labels[torch.randint(batch_split_size, (batch_split_size,)), :])
                        # labs_enc = discriminator(lat_top_dec, labels)
                        #
                        # loss_dis_unb = (1 / batch_split) * compute_gan_loss(labs_enc, model_manager.it % 2)
                        # model_manager.loss_backward(loss_dis_unb, nets_to_train)
                        # loss_dis_unb_sum += loss_dis_unb.item()

                    if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                        reg_dis_max = max(reg_dis_enc_sum, reg_dis_dec_sum)
                        loss_dis_min = min(loss_dis_enc_sum, loss_dis_dec_sum)
                        d_reg_every_mean = d_reg_every_mean_next
                        d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every,
                                                                                    d_reg_param_mean, 1 / d_reg_param,
                                                                                    reg_dis_max, reg_dis_target, loss_dis_min)

                    # g_factor_enc, g_factor_dec = update_g_factors(g_factor_enc, g_factor_dec, labs_dis_enc_sign, labs_dis_dec_sign, sign_mean_target)
                    # reg_dis_target = config['training']['lr'] * ((0.5 * (g_factor_enc + g_factor_dec)) ** 4)
                    # dis_encoder.fire_rate = 0.5 * (g_factor_enc + g_factor_dec)
                    # grad_mult(dis_encoder, 0.5 * (g_factor_enc + g_factor_dec))
                    # grad_mult(discriminator, 0.5 * (g_factor_enc + g_factor_dec))

                with model_manager.on_step(['encoder', 'decoder', 'generator']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)

                        z_enc, _, _ = encoder(images, labels)
                        lat_enc = generator(z_enc, labels)
                        # lat_enc = (1. - g_factor_enc) * lat_enc + g_factor_enc * torch.randn_like(lat_enc)
                        # images_dec, out_embs, _ = decoder(lat_enc.detach().clone())
                        # images_redec, _, _ = redecoder(lat_enc.detach().clone(), out_embs[-1])
                        #
                        # loss_dec = (1 / batch_mult) * F.mse_loss(images_redec, images)
                        # model_manager.loss_backward(loss_dec, nets_to_train, retain_graph=True)
                        # loss_dec_sum += loss_dec.item()

                        lat_top_enc, _, _ = dis_encoder(images, lat_enc)
                        labs_enc = discriminator(lat_top_enc)

                        if g_reg_every > 0 and model_manager.it % g_reg_every == 0:
                            reg_gen_enc, pl_mean_enc = compute_pl_reg(lat_enc, images, pl_mean_enc)
                            reg_gen_enc = (1 / batch_mult) * g_reg_every * reg_gen_enc
                            model_manager.loss_backward(reg_gen_enc, nets_to_train, retain_graph=True)
                            reg_gen_enc_sum += reg_gen_enc.item() / g_reg_every

                        loss_gen_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)
                        # labs_enc.register_hook(grad_mult_hook(g_factor_dec ** 0.5))
                        model_manager.loss_backward(loss_gen_enc, nets_to_train)
                        loss_gen_enc_sum += loss_gen_enc.item()

                        # images_dec, out_embs, _ = decoder(lat_enc.detach().clone())
                        # corrupt_inits = rand_circle_masks(out_embs[-1])
                        # images_redec, _, _ = decoder(lat_enc.detach().clone(), corrupt_inits[model_manager.it % 3])
                        #
                        # lat_top_redec, _, _ = dis_encoder(images_redec, lat_enc.detach().clone())
                        # labs_redec = discriminator(lat_top_redec, labels)
                        #
                        # loss_gen_redec = (1 / batch_mult) * 0.5 * compute_gan_loss(labs_redec, 1)
                        # model_manager.loss_backward(loss_gen_redec, nets_to_train)
                        # loss_gen_redec_sum += loss_gen_redec.item()

                        lat_gen = generator(z_gen, labels)
                        # lat_gen = (1. - g_factor_dec) * lat_gen + g_factor_dec * torch.randn_like(lat_gen)
                        images_dec, _, _ = decoder(lat_gen)

                        lat_top_dec, _, _ = dis_encoder(images_dec, lat_gen)
                        labs_dec = discriminator(lat_top_dec)

                        if g_reg_every > 0 and model_manager.it % g_reg_every == 0:
                            reg_gen_dec, pl_mean_dec = compute_pl_reg(images_dec, lat_gen, pl_mean_dec)
                            reg_gen_dec = (1 / batch_mult) * g_reg_every * reg_gen_dec
                            model_manager.loss_backward(reg_gen_dec, nets_to_train, retain_graph=True)
                            reg_gen_dec_sum += reg_gen_dec.item() / g_reg_every

                        loss_gen_dec = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                        # labs_dec.register_hook(grad_mult_hook(g_factor_enc ** 0.5))
                        model_manager.loss_backward(loss_gen_dec, nets_to_train)
                        loss_gen_dec_sum += loss_gen_dec.item()

                    # if isinstance(generator, torch.nn.DataParallel):
                    #     grad_ema_update(generator.module)
                    # else:
                    #     grad_ema_update(generator)

                    # grad_mult(encoder, (0.5 * (g_factor_enc + g_factor_dec)) ** 0.5)
                    # grad_mult(decoder, (0.5 * (g_factor_enc + g_factor_dec)) ** 0.5)
                    # grad_mult(generator, (0.5 * (g_factor_enc + g_factor_dec)) ** 0.5)

                # Streaming Images
                with torch.no_grad():
                    lat_gen = generator(z_test, labels_test)
                    images_gen, out_embs, _ = decoder(lat_gen)
                    # images_regen, _, _ = redecoder(lat_gen, out_embs[-1])
                    # images_gen = torch.cat([images_gen, images_regen], dim=3)

                stream_images(images_gen, config_name + '/gaen', config['training']['out_dir'] + '/gaen')

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_enc_sum + loss_gen_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_dis='%.2e' % (np.sum(running_loss_dis) / running_factor),
                              loss_gen='%.2e' % (np.sum(running_loss_gen) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                model_manager.log_scalar('losses',  'loss_dis_enc',  loss_dis_enc_sum)
                model_manager.log_scalar('losses',  'labs_dis_enc_sign',  labs_dis_enc_sign)
                model_manager.log_scalar('losses',  'labs_dis_dec_sign',  labs_dis_dec_sign)
                model_manager.log_scalar('losses',  'loss_dis_dec',  loss_dis_dec_sum)
                # model_manager.log_scalar('losses',  'loss_dis_unb',  loss_dis_unb_sum)
                model_manager.log_scalar('losses',  'loss_gen_enc',  loss_gen_enc_sum)
                model_manager.log_scalar('losses',  'loss_gen_dec',  loss_gen_dec_sum)
                model_manager.log_scalar('losses',  'loss_dec',  loss_dec_sum)

                # model_manager.log_scalar('regs',  'g_factor_enc',  g_factor_enc)
                # model_manager.log_scalar('regs',  'g_factor_dec',  g_factor_dec)
                model_manager.log_scalar('regs',  'reg_dis_enc',  reg_dis_enc_sum)
                model_manager.log_scalar('regs',  'reg_dis_dec',  reg_dis_dec_sum)
                model_manager.log_scalar('regs',  'd_reg_every_mean',  d_reg_every_mean)
                model_manager.log_scalar('regs',  'd_reg_param_mean',  d_reg_param_mean)

                if g_reg_every > 0:
                    model_manager.log_scalar('regs',  'reg_gen_enc',  reg_gen_enc_sum)
                    model_manager.log_scalar('regs',  'reg_gen_dec',  reg_gen_dec_sum)

                    model_manager.log_scalar('regs',  'pl_mean_enc',  pl_mean_enc)
                    model_manager.log_scalar('regs',  'pl_mean_dec',  pl_mean_dec)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((model_manager.epoch + 1) % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, _, trainiter = get_inputs(trainiter, batch_size, device)
            lat_gen = generator(z_test, labels_test)
            images_gen, out_embs, _ = decoder(lat_gen)
            # images_regen, _, _ = redecoder(lat_gen, out_embs[-1])
            # images_gen = torch.cat([images_gen, images_regen], dim=3)
            z_enc, _, _ = encoder(images, labels)
            lat_enc = generator(z_enc, labels)
            images_dec, out_embs, _ = decoder(lat_enc)
            # images_redec, _, _ = redecoder(lat_enc, out_embs[-1])
            # images_dec = torch.cat([images_dec, images_redec], dim=3)
            model_manager.log_images(images,  'all_input')
            model_manager.log_images(images_gen,  'all_gen')
            model_manager.log_images(images_dec,  'all_dec')
            for lab in range(config['training']['sample_labels']):
                if labels_test.dim() == 1:
                    fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                else:
                    fixed_lab = labels_test.clone()
                    fixed_lab[:, lab] = 1
                lat_gen = generator(z_test, fixed_lab)
                images_gen, out_embs, _ = decoder(lat_gen)
                # images_regen, _, _ = redecoder(lat_gen, out_embs[-1])
                # images_gen = torch.cat([images_gen, images_regen], dim=3)
                model_manager.log_images(images_gen,  'class_%04d' % lab)

        # Perform inception
        if config['training']['inception_every'] > 0 and ((epoch + 1) % config['training']['inception_every']) == 0 and epoch > 0:
            t.write('Computing inception/fid!')
            inception_mean, inception_std, fid = compute_inception_score(generator, decoder,
                                                                         10000, 10000, config['training']['batch_size'],
                                                                         zdist, ydist, fid_real_samples, device)
            model_manager.log_scalar('inception_score',  'mean',  inception_mean)
            model_manager.log_scalar('inception_score',  'stddev',  inception_std)
            model_manager.log_scalar('inception_score',  'fid',  fid)

model_manager.save()
print('Training is complete...')
