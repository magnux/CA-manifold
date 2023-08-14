# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from torch.nn.utils import clip_grad_norm_
from tqdm import trange
from src.config import load_config
from src.distributions import get_ydist, get_zdist
from src.inputs import get_dataset
from src.utils.loss_utils import age_gaussian_kl_loss, compute_grad_reg, update_reg_params
from src.utils.model_utils import compute_inception_score, get_grad_norm
from src.utils.media_utils import rand_circle_masks, tensor_jamm, mask_templates
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images, video_fps
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a RAGE')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
n_calls = config['network']['kwargs']['n_calls']
n_epochs = config['training']['n_epochs']
d_reg_param = config['training']['d_reg_param']
d_reg_every = config['training']['d_reg_every']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
batch_mult_steps = config['training']['batch_mult_steps']
n_workers = config['training']['n_workers']
pre_train = config['training']['pre_train'] if 'pre_train' in config['training'] else False
kl_factor = config['training']['kl_factor'] if 'kl_factor' in config['training'] else 0.1

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
# zdist = get_zdist(config['z_dist']['type'], config['z_dist']['z_dim'] * 4, device=device)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    # 'generator': {'class': 'base', 'sub_class': 'Generator'},
    # 'lat_expander': {'class': 'base', 'sub_class': 'IRMExpander'},
    # 'lat_reducer': {'class': 'base', 'sub_class': 'IRMReducer'},
}
# to_avg = ['encoder', 'decoder', 'generator']

model_manager = ModelManager('rage', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
# generator = model_manager.get_network('generator')
# lat_expander = model_manager.get_network('lat_expander')
# lat_reducer = model_manager.get_network('lat_reducer')

# encoder_avg = model_manager.get_network_avg('encoder')
# decoder_avg = model_manager.get_network_avg('decoder')
# generator_avg = model_manager.get_network_avg('generator')

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
    # z_gen = zdist.sample((images.size(0),))
    # z_gen = F.normalize(z_gen)
    # z_gen.detach_().requires_grad_()
    return images, labels, trainiter


images_test, labels_test, trainiter = get_inputs(iter(trainloader), batch_size, device)


if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()

window_size = math.ceil((len(trainloader) // batch_split) / 10)

# if pre_train:
#     for epoch in range(model_manager.epoch, n_epochs // 64):
#         with model_manager.on_epoch():
#             running_loss_dec = np.zeros(window_size)
#
#             t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
#             t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
#             for batch in t:
#                 with model_manager.on_batch():
#
#                     loss_dis_unb_sum = 0
#                     loss_gen_unb_sum = 0
#
#                     with model_manager.on_step(['encoder', 'decoder', 'generator', 'lat_expander', 'lat_reducer']) as nets_to_train:
#                         for _ in range(batch_split):
#                             images, labels, z_gen, trainiter = get_inputs(trainiter, batch_split_size, device)
#
#                             z_enc, _, _ = encoder(images, labels)
#                             z_enc = lat_expander(z_enc)
#
#                             loss_dis_unb = (1 / batch_split) * age_gaussian_kl_loss(z_enc) * -1 ** (model_manager.it % 2)
#                             model_manager.loss_backward(loss_dis_unb, nets_to_train)
#                             loss_dis_unb_sum += loss_dis_unb.item()
#
#                             z_gen_red = lat_reducer(z_gen)
#                             lat_gen = generator(z_gen_red, labels)
#                             images_dec, out_embs, _ = decoder(lat_gen)
#                             images_redec, _, _ = decoder(lat_gen, out_embs[-1])
#                             z_redec, _, _ = encoder(images_redec, labels)
#                             z_redec = lat_expander(z_redec)
#
#                             loss_gen_unb = (1 / batch_split) * age_gaussian_kl_loss(z_redec) * -1 ** ((it + 1) % 2)
#                             model_manager.loss_backward(loss_gen_unb, nets_to_train)
#                             loss_gen_unb_sum += loss_gen_unb.item()
#
#                 # Print progress
#                 running_loss_dec[batch % window_size] = loss_dis_unb_sum + loss_gen_unb_sum
#                 running_factor = window_size if batch > window_size else batch + 1
#                 t.set_postfix(loss_unbias='%.2e' % (np.sum(running_loss_dec) / running_factor))
#
#                 # Log progress
#                 model_manager.log_scalar('losses_pretrain',  'loss_dis_unb',  loss_dis_unb_sum)
#                 model_manager.log_scalar('losses_pretrain',  'loss_gen_unb',  loss_gen_unb_sum)
#
#     print('Pre-training is complete...')
#     model_manager.epoch = max(model_manager.epoch, n_epochs // 64)


train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
print('Starting training, phase: %d' % train_phase)
mtemps = mask_templates(image_size)  # train_phase)


d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', d_reg_every if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)

base_perm = np.arange(0, batch_split_size)
prev_perm = base_perm.copy()
next_perm = None

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split
        # Discriminator reg target
        reg_dis_target = config['training']['lr']  # 1. * (1. - 0.999 ** (n_epochs / (epoch + 1e-8)))
        # Discriminator mean sign target
        sign_mean_target = 0.2  # 0.5 * (1. - 0.9 ** (n_epochs / (epoch + 1e-8)))

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum, loss_dis_redec_sum = 0, 0, 0
                # reg_dis_enc_sum, reg_dis_dec_sum = 0, 0
                loss_gen_enc_sum, loss_gen_dec_sum, loss_gen_redec_sum = 0, 0, 0
                # loss_dec_sum = 0

                # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                #     d_reg_factor = (d_reg_every_mean_next - (model_manager.it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                # else:
                #     reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                #     reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                if next_perm is None:
                    next_perm = prev_perm.copy()
                    np.random.shuffle(next_perm)

                # Discriminator step
                with model_manager.on_step(['encoder']) as nets_to_train:

                    for _ in range(batch_mult):
                        images, labels, trainiter = get_inputs(trainiter, batch_split_size, device)

                        z_enc, _, _ = encoder(images, labels)
                        # z_enc = lat_expander(z_enc)
                        z_enc = F.normalize(z_enc)

                        # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                        #     reg_dis_enc = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_enc, images)
                        #     model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                        #     reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor
                        #
                        #     # reg_dis_enc = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_enc, encoder.inj_lat)
                        #     # model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                        #     # reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                        loss_dis_enc = (1 / batch_mult) * kl_factor * age_gaussian_kl_loss(z_enc)
                        model_manager.loss_backward(loss_dis_enc, nets_to_train)
                        loss_dis_enc_sum += loss_dis_enc.item()

                        with torch.no_grad():
                            z_enc, _, _ = encoder(images, labels)
                            # z_enc = lat_expander(z_enc)
                            z_enc = F.normalize(z_enc)

                            fussion_progress = 0
                            if model_manager.it % 2 == 1:
                                z_enc_next, _, _ = encoder(images[next_perm], labels[next_perm])
                                z_enc_next = F.normalize(z_enc_next)
                                fussion_progress = ((model_manager.it % video_fps) / video_fps)
                                z_enc = (1 - fussion_progress) * z_enc + fussion_progress * z_enc_next
                            images_dec, out_embs, _ = decoder(z_enc)
                            if model_manager.it % 2 == 0:
                                images_dec, _ = tensor_jamm(images, images_dec, mtemps, True)

                            images_redec, _, _ = decoder(z_enc, out_embs[-1])
                            if model_manager.it % 2 == 0:
                                images_redec, _ = tensor_jamm(images, images_redec, mtemps, True)

                        images_dec.requires_grad_()
                        images_redec.requires_grad_()

                        z_dec, _, _ = encoder(images_dec, labels[next_perm] if fussion_progress > 0.5 else labels)
                        # z_dec = lat_expander(z_dec)
                        z_dec = F.normalize(z_dec)

                        # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                        #     reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_redec, images_redec)
                        #     model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                        #     reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor
                        #
                        #     # reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_redec, encoder.inj_lat)
                        #     # model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                        #     # reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                        loss_dis_dec = (1 / batch_mult) * 0.9 * kl_factor * -age_gaussian_kl_loss(z_dec)
                        model_manager.loss_backward(loss_dis_dec, nets_to_train)
                        loss_dis_dec_sum -= loss_dis_dec.item()


                        z_redec, _, _ = encoder(images_redec, labels[next_perm] if fussion_progress > 0.5 else labels)
                        # z_redec = lat_expander(z_redec)
                        z_redec = F.normalize(z_redec)

                        # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                        #     reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_redec, images_redec)
                        #     model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                        #     reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor
                        #
                        #     # reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(z_redec, encoder.inj_lat)
                        #     # model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                        #     # reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                        loss_dis_redec = (1 / batch_mult) * 0.1 * kl_factor * -age_gaussian_kl_loss(z_redec)
                        model_manager.loss_backward(loss_dis_redec, nets_to_train)
                        loss_dis_redec_sum -= loss_dis_redec.item()

                # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                #         reg_dis_max = max(reg_dis_enc_sum, reg_dis_dec_sum)
                #         loss_dis_min = min(loss_dis_enc_sum, loss_dis_dec_sum)
                #         d_reg_every_mean = d_reg_every_mean_next
                #         d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every,
                #                                                                     d_reg_param_mean, 1 / d_reg_param,
                #                                                                     reg_dis_max, reg_dis_target, loss_dis_min)

                for _ in range(4):
                    # Generator step
                    with model_manager.on_step(['decoder', 'generator']) as nets_to_train:

                        for _ in range(batch_mult):
                            images, labels, trainiter = get_inputs(trainiter, batch_split_size, device)

                            z_enc, _, _ = encoder(images, labels)
                            # z_enc = lat_expander(z_enc)
                            z_enc = F.normalize(z_enc)

                            fussion_progress = 0
                            if model_manager.it % 2 == 1:
                                z_enc_next, _, _ = encoder(images[next_perm], labels[next_perm])
                                z_enc_next = F.normalize(z_enc_next)
                                fussion_progress = ((model_manager.it % video_fps) / video_fps)
                                z_enc = (1 - fussion_progress) * z_enc + fussion_progress * z_enc_next
                            images_dec, out_embs, _ = decoder(z_enc)
                            if model_manager.it % 2 == 0:
                                images_dec, _ = tensor_jamm(images, images_dec, mtemps, True)

                            z_dec, _, _ = encoder(images_dec, labels[next_perm] if fussion_progress > 0.5 else labels)
                            # z_dec = lat_expander(z_dec)
                            z_dec = F.normalize(z_dec)

                            loss_gen_dec = (1 / batch_mult) * 0.9 * kl_factor * age_gaussian_kl_loss(z_dec)
                            model_manager.loss_backward(loss_gen_dec, nets_to_train, retain_graph=True)
                            loss_gen_dec_sum += loss_gen_dec.item()

                            corrupt_inits = rand_circle_masks(out_embs[-1])
                            images_redec, _, _ = decoder(z_enc, corrupt_inits[model_manager.it % 3])
                            if model_manager.it % 2 == 0:
                                images_redec, _ = tensor_jamm(images, images_redec, mtemps, True)

                            z_redec, _, _ = encoder(images_redec, labels[next_perm] if fussion_progress > 0.5 else labels)
                            z_redec = F.normalize(z_redec)

                            loss_gen_redec = (1 / batch_mult) * 0.1 * kl_factor * age_gaussian_kl_loss(z_redec)
                            model_manager.loss_backward(loss_gen_redec, nets_to_train)
                            loss_gen_redec_sum += loss_gen_redec.item()

                            # z_enc, _, _ = encoder(images, labels)
                            # z_enc = lat_expander(z_enc)
                            # z_enc = F.normalize(z_enc)
                            # z_enc_red = lat_reducer(z_enc)
                            # lat_dec = generator(z_enc_red, labels)
                            # images_dec, _, _ = decoder(lat_dec)
                            # z_dec, _, _ = encoder(images_dec, labels)
                            # z_dec = lat_expander(z_dec)
                            # z_dec = F.normalize(z_dec)

                            # loss_dec = (1 / batch_mult) * F.mse_loss(z_dec, z_enc.detach())
                            # # loss_dec = (1 / batch_mult) * (2 - z_dec.mul(z_enc).mean())
                            # model_manager.loss_backward(loss_dec, nets_to_train)
                            # loss_dec_sum += loss_dec.item()

                # Streaming Images
                with torch.no_grad():
                    images_gen_l = []
                    for i in range(batch_split):
                        ini = i * batch_split_size
                        end = (i + 1) * batch_split_size
                        
                        z_enc, _, _ = encoder(images_test[ini:end][prev_perm], labels_test[ini:end][prev_perm])
                        z_enc = F.normalize(z_enc)

                        z_enc_next, _, _ = encoder(images_test[ini:end][next_perm], labels_test[ini:end][next_perm])
                        z_enc_next = F.normalize(z_enc_next)
                        fussion_progress = ((model_manager.it % video_fps) / video_fps)
                        z_enc = (1 - fussion_progress) * z_enc + fussion_progress * z_enc_next

                        images_gen, out_embs, _ = decoder(z_enc)
                        # images_regen, _, _ = decoder(z_enc, out_embs[-1])
                        images_gen_l.append(images_gen)
                    images_gen = torch.cat(images_gen_l)
                    # images_gen = torch.cat([images_test, images_gen], dim=3)

                stream_images(images_gen, config_name + '/rage', config['training']['out_dir'] + '/rage')

                if (model_manager.it + 1) % video_fps == 0:
                    prev_perm = next_perm
                    next_perm = None

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
                model_manager.log_scalar('losses',  'loss_dis_dec',  loss_dis_dec_sum)
                model_manager.log_scalar('losses', 'loss_dis_redec', loss_dis_redec_sum)

                model_manager.log_scalar('losses',  'loss_gen_dec',  loss_gen_dec_sum)
                model_manager.log_scalar('losses', 'loss_gen_redec', loss_gen_redec_sum)

                # model_manager.log_scalar('losses',  'loss_dec',  loss_dec_sum)

                # model_manager.log_scalar('regs',  'reg_dis_enc',  reg_dis_enc_sum)
                # model_manager.log_scalar('regs',  'reg_dis_dec',  reg_dis_dec_sum)
                # model_manager.log_scalar('regs',  'd_reg_every_mean',  d_reg_every_mean)
                # model_manager.log_scalar('regs',  'd_reg_param_mean',  d_reg_param_mean)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((model_manager.epoch + 1) % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, trainiter = get_inputs(trainiter, batch_size, device)
            images_gen_l = []
            images_dec_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size

                z_enc, _, _ = encoder(images_test[ini:end], labels_test[ini:end])
                z_enc = F.normalize(z_enc)
                images_gen, out_embs, _ = decoder(z_enc)
                images_regen, _, _ = decoder(z_enc, out_embs[-1])
                images_gen = torch.cat([images_test[ini:end], images_gen, images_regen], dim=3)
                images_gen_l.append(images_gen)

                z_enc, _, _ = encoder(images[ini:end], labels[ini:end])
                z_enc = F.normalize(z_enc)
                images_dec, out_embs, _ = decoder(z_enc)
                images_redec, _, _ = decoder(z_enc, out_embs[-1])
                images_dec = torch.cat([images[ini:end], images_dec, images_redec], dim=3)
                images_dec_l.append(images_dec)
            images_gen = torch.cat(images_gen_l)
            images_dec = torch.cat(images_dec_l)
            model_manager.log_images(images_gen,  'fixed_samples')
            model_manager.log_images(images_dec,  'random_samples')
            # for lab in range(config['training']['sample_labels']):
            #     if labels_test.dim() == 1:
            #         fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
            #     else:
            #         fixed_lab = labels_test.clone()
            #         fixed_lab[:, lab] = 1
            #     z_enc = generator(z_test, fixed_lab)
            #     images_gen, out_embs, _ = decoder(z_enc)
            #     # images_regen, _, _ = redecoder(z_enc, out_embs[-1])
            #     # images_gen = torch.cat([images_gen, images_regen], dim=3)
            #     model_manager.log_images(images_gen,  'class_%04d' % lab)

        # # Perform inception
        # if config['training']['inception_every'] > 0 and ((model_manager.epoch + 1) % config['training']['inception_every']) == 0 and model_manager.epoch > 0:
        #     t.write('Computing inception/fid!')
        #     inception_mean, inception_std, fid = compute_inception_score(generator, decoder,
        #                                                                  10000, 10000, config['training']['batch_size'],
        #                                                                  zdist, ydist, fid_real_samples, device)
        #     model_manager.log_scalar('inception_score',  'mean',  inception_mean)
        #     model_manager.log_scalar('inception_score',  'stddev',  inception_std)
        #     model_manager.log_scalar('inception_score',  'fid',  fid)

model_manager.save()
print('Training is complete...')
