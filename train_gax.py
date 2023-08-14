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
from src.utils.loss_utils import compute_gan_loss, compute_grad_reg, compute_pl_reg, update_reg_params, update_g_factors, CE_ZERO, compute_dir_grad_reg
from src.utils.model_utils import compute_inception_score, grad_mult, grad_mult_hook, grad_dither, grad_dither_hook, update_network_average, grad_ema_update
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a GAX')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

# config['network']['kwargs']['norm_z'] = True
# config['network']['kwargs']['irm_z'] = True
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
z_dim = config['z_dist']['z_dim']
lat_size = lat_size = config['network']['kwargs']['lat_size']
pre_train = config['training']['pre_train'] if 'pre_train' in config['training'] else False

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
if config['data']['name'] == 'emoji':
    sampler = trainset.get_equalized_class_sampler()
else:
    samples_weight = torch.ones(len(trainset)) / len(trainset)
    sampler = torch.utils.data.WeightedRandomSampler(samples_weight, len(samples_weight))
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True if sampler is None else False,
                                          num_workers=n_workers, drop_last=True, sampler=sampler)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Distributions
ydist = get_ydist(config['data']['n_labels'], device=device)
zdist = get_zdist(config['z_dist']['type'], z_dim, device=device)

# Networks
networks_dict = {
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
    'generator': {'class': 'base', 'sub_class': 'Generator'},
    'dis_encoder': {'class': config['network']['class'], 'sub_class': 'LabsInjectedEncoder'},
    'discriminator': {'class': 'base', 'sub_class': 'UnconditionalDiscriminator'},
}
# to_avg = ['decoder', 'generator']

model_manager = ModelManager('gax', networks_dict, config)
decoder = model_manager.get_network('decoder')
generator = model_manager.get_network('generator')
dis_encoder = model_manager.get_network('dis_encoder')
discriminator = model_manager.get_network('discriminator')

# decoder_avg = model_manager.get_network_avg('decoder')
# generator_avg = model_manager.get_network_avg('generator')

model_manager.print()

embedding_mat = torch.eye(n_labels, device=device)

def get_inputs(trainiter, batch_size, device):
    images, labels, idxes = [], [], []
    n_batches = math.ceil(batch_size / batch_split_size)
    for _ in range(n_batches):
        next_inputs = next(trainiter, None)
        if trainiter is None or next_inputs is None:
            trainiter = iter(trainloader)
            next_inputs = next(trainiter, None)
        images.append(next_inputs[0])
        labels.append(next_inputs[1])
        if not sampler is None:
            idxes.extend(next_inputs[2].numpy().tolist())
    images = torch.cat(images, 0)
    labels = torch.cat(labels, 0)
    if batch_size % config['training']['batch_size'] > 0:
        images, labels = images[:batch_size, ...], labels[:batch_size, ...]
        if not sampler is None:
            idxes = idxes[:batch_size]
    images, labels = images.to(device), labels.to(device)
    if labels.dtype is torch.int64:
        if labels.dim() == 1:
            labels = embedding_mat[labels]
        else:
            labels = labels.to(torch.float32)
    images = images.detach().requires_grad_()
    z_gen = zdist.sample((images.size(0),))
    z_gen.detach_().requires_grad_()
    return images, labels, z_gen, trainiter, idxes


images_test, labels_test, z_test, trainiter, _ = get_inputs(iter(trainloader), batch_size, device)
images_test_cropped, images_test_mushed, images_test_shuffled, images_test_mask = rand_circle_masks(images_test)

if config['training']['inception_every'] > 0:
    fid_real_samples = []
    for _ in range(10000 // batch_size):
        images, _, _, trainiter, _ = get_inputs(trainiter, batch_size, torch.device('cpu'))
        fid_real_samples.append(images)
    fid_real_samples = torch.cat(fid_real_samples, dim=0)[:10000, ...].detach().numpy()

window_size = math.ceil((len(trainloader) // batch_split) / 10)

if pre_train:
    for epoch in range(model_manager.epoch, n_epochs // 64):
        with model_manager.on_epoch():
            running_loss_dec = np.zeros(window_size)

            t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
            t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
            for batch in t:
                with model_manager.on_batch():

                    loss_dis_unb_sum = 0
                    loss_gen_unb_sum = 0

                    with model_manager.on_step(['dis_encoder', 'discriminator', 'decoder', 'generator']) as nets_to_train:
                        for _ in range(batch_split):
                            images, labels, z_gen, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                            lat_top_enc, _, _ = dis_encoder(images, labels)
                            labs_enc = discriminator(lat_top_enc)

                            loss_dis_unb = (1 / batch_split) * compute_gan_loss(labs_enc, model_manager.it % 2)
                            model_manager.loss_backward(loss_dis_unb, nets_to_train)
                            loss_dis_unb_sum += loss_dis_unb.item()

                            lat_gen = generator(z_gen, labels)
                            images_dec, _, _ = decoder(lat_gen)

                            lat_top_dec, _, _ = dis_encoder(images_dec, labels)
                            labs_dec = discriminator(lat_top_dec)

                            loss_gen_unb = (1 / batch_split) * compute_gan_loss(labs_dec, (model_manager.it+1) % 2)
                            model_manager.loss_backward(loss_gen_unb, nets_to_train)
                            loss_gen_unb_sum += loss_gen_unb.item()

                # Print progress
                running_loss_dec[batch % window_size] = loss_dis_unb_sum + loss_gen_unb_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss_unbias='%.2e' % (np.sum(running_loss_dec) / running_factor))

                # Log progress
                model_manager.log_scalar('losses_pretrain',  'loss_dis_unb',  loss_dis_unb_sum)
                model_manager.log_scalar('losses_pretrain',  'loss_gen_unb',  loss_gen_unb_sum)

    print('Pre-training is complete...')
    model_manager.epoch = max(model_manager.epoch, n_epochs // 64)


d_reg_every_mean = model_manager.log_manager.get_last('regs', 'd_reg_every_mean', d_reg_every if d_reg_every > 0 else 0)
d_reg_every_mean_next = d_reg_every_mean
d_reg_param_mean = model_manager.log_manager.get_last('regs', 'd_reg_param_mean', 1 / d_reg_param)
# d_unb_param_mean = model_manager.log_manager.get_last('regs', 'd_unb_param_mean', 1 / d_reg_param)

pl_mean_dec = model_manager.log_manager.get_last('regs', 'pl_mean_dec', 0.)

torch.autograd.set_detect_anomaly(True)
g_factor_enc = model_manager.log_manager.get_last('regs', 'g_factor_enc', 1e-3)
g_factor_dec = model_manager.log_manager.get_last('regs', 'g_factor_dec', 1e-3)

labs_dis_enc_sign_mm = model_manager.log_manager.get_last('regs', 'labs_dis_enc_sign_mm', 0.)
labs_dis_dec_sign_mm = model_manager.log_manager.get_last('regs', 'labs_dis_dec_sign_mm', 0.)

train_phase = model_manager.epoch // n_epochs
n_epochs *= 1 + train_phase
print('Starting training, phase: %d' % train_phase)

if not sampler is None:
    new_weights = trainloader.sampler.weights.clone()
else:
    new_weights = None
for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss_dis = np.zeros(window_size)
        running_loss_gen = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split
        # Discriminator reg target
        # if train_phase == 2:
        #     reg_dis_target = config['training']['lr'] + (0.999 ** ((n_epochs // 3) / ((epoch - (n_epochs // 3) * 2) + 1e-8)))
        # else:
        reg_dis_target = 1e-7 * (0.7 ** train_phase)
        # if train_phase >= 2:
        #     reg_turns = np.random.rand((n_epochs // 3) * (len(trainloader) // batch_split)) > 0.5
        #     p_s_it = (n_epochs // (train_phase + 1)) * train_phase * (len(trainloader) // batch_split)
        #     if train_phase == 2:
        #         s = 0
        #         for _ in range(5):
        #             half_s = int(0.15 * (((n_epochs // 3) * (len(trainloader) // batch_split)) - s))
        #             next_s = int(0.3 * (((n_epochs // 3) * (len(trainloader) // batch_split)) - s))
        #             reg_turns[s:s + half_s] = False
        #             reg_turns[s + half_s:s + next_s] = True
        #             s = s + next_s
        # Discriminator mean sign target
        sign_mean_target = 0.2  # 0.5 * (1. - 0.9 ** (n_epochs / (epoch + 1e-8)))

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            if not sampler is None and model_manager.it % 100 == 99:
                trainloader.sampler.weights = new_weights
                with torch.no_grad():
                    new_weights = new_weights.clone()
                    new_weights += new_weights.mean()
                    new_weights /= 2.

            with model_manager.on_batch():

                loss_dis_enc_sum, loss_dis_dec_sum, loss_dis_redec_sum = 0, 0, 0
                labs_dis_enc_sign, labs_dis_dec_sign = 0, 0
                loss_gen_dec_sum, loss_gen_redec_sum = 0, 0

                reg_dis_enc_sum, reg_dis_dec_sum = 0, 0
                reg_gen_dec_sum, reg_dis_unb_sum = 0, 0

                if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                    d_reg_factor = (d_reg_every_mean_next - (model_manager.it % d_reg_every_mean_next)) * (1 / d_reg_param_mean)
                else:
                    reg_dis_enc_sum = model_manager.log_manager.get_last('regs', 'reg_dis_enc')
                    reg_dis_dec_sum = model_manager.log_manager.get_last('regs', 'reg_dis_dec')

                # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == d_reg_every_mean // 2:
                #     d_unb_factor = (d_reg_every_mean_next - (model_manager.it % d_reg_every_mean_next)) * (1 / d_unb_param_mean)
                # else:
                #     reg_dis_unb_sum = model_manager.log_manager.get_last('regs', 'reg_dis_unb')

                if not (g_reg_every > 0 and model_manager.it % g_reg_every == 1):
                    reg_gen_dec_sum = model_manager.log_manager.get_last('regs', 'reg_gen_dec')

                def train_d(loss_dis_enc_sum, labs_dis_enc_sign, labs_dis_enc_sign_mm, reg_dis_enc_sum, new_weights,
                            loss_dis_dec_sum, labs_dis_dec_sign, labs_dis_dec_sign_mm, reg_dis_dec_sum, g_factor_enc, g_factor_dec,
                            loss_dis_redec_sum, trainiter, d_reg_param_mean, d_reg_every_mean, d_reg_every_mean_next):

                    # Discriminator step
                    with model_manager.on_step(['dis_encoder', 'discriminator']) as nets_to_train:

                        for _ in range(batch_mult):
                            images, labels, z_gen, trainiter, idxes = get_inputs(trainiter, batch_split_size, device)

                            lat_top_enc, out_embs_enc, _ = dis_encoder(images, labels)
                            # lat_top_enc = (1. - g_factor_enc) * lat_top_enc + g_factor_enc * torch.randn_like(lat_top_enc)
                            if not sampler is None:
                                with torch.no_grad():
                                    inv_acts = 0.
                                    for out_emb in out_embs_enc:
                                        inv_acts_max = out_emb.max(3, keepdim=True)[0].max(2, keepdim=True)[0]
                                        inv_acts_min = out_emb.min(3, keepdim=True)[0].min(2, keepdim=True)[0]
                                        inv_acts_tmp = (out_emb - inv_acts_min) / (inv_acts_max - inv_acts_min)
                                        inv_acts += 1 / (inv_acts_tmp.mean(dim=(1, 2, 3)) + 1e-4).to('cpu')
                                    inv_acts /= len(out_embs_enc)
                                    new_weights[idxes] = (new_weights[idxes] + inv_acts) / 2
                            labs_enc = discriminator(lat_top_enc)
                            labs_dis_enc_sign += ((1 / batch_mult) * labs_enc.sign().mean()).item()

                            # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0 and (train_phase == 0 or (train_phase >= 2 and not reg_turns[it - p_s_it])):
                            #     reg_dis_enc = (1 / batch_mult) * d_reg_factor * compute_grad_reg(labs_enc, images)
                            #     model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            #     reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor
                            #
                            #     # reg_dis_enc = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_enc, dis_encoder.inj_lat)
                            #     # model_manager.loss_backward(reg_dis_enc, nets_to_train, retain_graph=True)
                            #     # reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                            loss_dis_enc = (1 / batch_mult) * compute_gan_loss(labs_enc, 1)
                            # labs_enc.register_hook(grad_mult_hook(g_factor_enc))
                            model_manager.loss_backward(loss_dis_enc, nets_to_train, retain_graph=True)
                            loss_dis_enc_sum += loss_dis_enc.item()

                            if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                                loss_dis_enc_inv = (1 / batch_mult) * compute_gan_loss(labs_enc, 0)
                                reg_dis_enc = d_reg_factor * compute_dir_grad_reg(loss_dis_enc, loss_dis_enc_inv, images)
                                model_manager.loss_backward(reg_dis_enc, nets_to_train)
                                reg_dis_enc_sum += reg_dis_enc.item() / d_reg_factor

                            with torch.no_grad():
                                lat_gen = generator(z_gen, labels)
                                # lat_gen = (1. - g_factor_dec) * lat_gen + g_factor_dec * torch.randn_like(lat_gen)
                                images_dec, out_embs, _ = decoder(lat_gen)
                                corrupt_inits = rand_circle_masks(out_embs[-1])
                                images_redec, _, _ = decoder(lat_gen, corrupt_inits[model_manager.it % 3])

                            images_dec.requires_grad_()
                            images_redec.requires_grad_()
                            lat_top_dec, out_embs_dec, _ = dis_encoder(images_dec, labels)
                            # lat_top_dec = (1. - g_factor_dec) * lat_top_dec + g_factor_dec * torch.randn_like(lat_top_dec)
                            labs_dec = discriminator(lat_top_dec)
                            labs_dis_dec_sign -= ((1 / batch_mult) * labs_dec.sign().mean()).item()

                            # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0 and (train_phase <= 1 or (train_phase >= 2 and reg_turns[it - p_s_it])):
                            #     reg_dis_dec = (1 / batch_mult) * d_reg_factor * compute_grad_reg(labs_dec, images_dec)
                            #     model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            #     reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor
                            #
                            #     # reg_dis_dec = (1 / batch_mult) * 0.5 * d_reg_factor * compute_grad_reg(labs_dec, dis_encoder.inj_lat)
                            #     # model_manager.loss_backward(reg_dis_dec, nets_to_train, retain_graph=True)
                            #     # reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                            loss_dis_dec = (1 / batch_mult) * 0.5 * compute_gan_loss(labs_dec, 0)
                            # labs_dec.register_hook(grad_mult_hook(g_factor_dec))
                            model_manager.loss_backward(loss_dis_dec, nets_to_train, retain_graph=True)
                            loss_dis_dec_sum += loss_dis_dec.item()

                            if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                                loss_dis_dec_inv = (1 / batch_mult) * compute_gan_loss(labs_dec, 1)
                                reg_dis_dec = d_reg_factor * compute_dir_grad_reg(loss_dis_dec, loss_dis_dec_inv, images_dec)
                                model_manager.loss_backward(reg_dis_dec, nets_to_train)
                                reg_dis_dec_sum += reg_dis_dec.item() / d_reg_factor

                            lat_top_redec, _, _ = dis_encoder(images_redec, labels)
                            labs_redec = discriminator(lat_top_redec)

                            loss_dis_redec = (1 / batch_mult) * 0.5 * compute_gan_loss(labs_redec, 0)
                            model_manager.loss_backward(loss_dis_redec, nets_to_train)
                            loss_dis_redec_sum += loss_dis_redec.item()

                        labs_dis_enc_sign_mm = 0.99 * labs_dis_enc_sign_mm + 0.01 * labs_dis_enc_sign
                        labs_dis_dec_sign_mm = 0.99 * labs_dis_dec_sign_mm + 0.01 * labs_dis_dec_sign
                        if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == 0:
                            reg_dis_max = max(reg_dis_enc_sum, reg_dis_dec_sum)
                            loss_dis_min = min(loss_dis_enc_sum, loss_dis_dec_sum)
                            d_reg_every_mean = d_reg_every_mean_next
                            d_reg_every_mean_next, d_reg_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every,
                                                                                        d_reg_param_mean, 1 / d_reg_param,
                                                                                        reg_dis_max, reg_dis_target, loss_dis_min)

                        # if d_reg_every_mean > 0 and model_manager.it % d_reg_every_mean == d_reg_every_mean // 2:
                        #     _, d_unb_param_mean = update_reg_params(d_reg_every_mean_next, d_reg_every,
                        #                                             d_unb_param_mean, 1 / d_reg_param,
                        #                                             reg_dis_unb_sum, reg_dis_target, None, update_every=False)

                        # g_factor_enc, g_factor_dec = update_g_factors(g_factor_enc, g_factor_dec, labs_dis_enc_sign, labs_dis_dec_sign, sign_mean_target)
                        # reg_dis_target = config['training']['lr'] * ((0.5 * (g_factor_enc + g_factor_dec)) ** 4)
                        # dis_encoder.fire_rate = 0.5 * (g_factor_enc + g_factor_dec)
                        # grad_mult(dis_encoder, 0.5 * (g_factor_enc + g_factor_dec))
                        # grad_mult(discriminator, 0.5 * (g_factor_enc + g_factor_dec))

                    return (loss_dis_enc_sum, labs_dis_enc_sign, labs_dis_enc_sign_mm, reg_dis_enc_sum, new_weights,
                            loss_dis_dec_sum, labs_dis_dec_sign, labs_dis_dec_sign_mm, reg_dis_dec_sum, g_factor_enc, g_factor_dec,
                            loss_dis_redec_sum, trainiter, d_reg_param_mean, d_reg_every_mean, d_reg_every_mean_next)

                def train_g(loss_gen_dec_sum, loss_gen_redec_sum, reg_gen_dec_sum, trainiter, pl_mean_dec):
                    # Generator step
                    with model_manager.on_step(['decoder', 'generator']) as nets_to_train:

                        for _ in range(batch_mult):
                            images, labels, z_gen, trainiter, _ = get_inputs(trainiter, batch_split_size, device)

                            lat_gen = generator(z_gen, labels)
                            # lat_gen = (1. - g_factor_dec) * lat_gen + g_factor_dec * torch.randn_like(lat_gen)
                            images_dec, out_embs, _ = decoder(lat_gen)

                            lat_top_dec, _, _ = dis_encoder(images_dec, labels)
                            # lat_top_dec = (1. - g_factor_dec) * lat_top_dec + g_factor_dec * torch.randn_like(lat_top_dec)
                            labs_dec = discriminator(lat_top_dec)

                            if g_reg_every > 0 and model_manager.it % g_reg_every == 0:
                                reg_gen_dec, pl_mean_dec = compute_pl_reg(images_dec, lat_gen, pl_mean_dec)
                                reg_gen_dec = (1 / batch_mult) * g_reg_every * reg_gen_dec
                                model_manager.loss_backward(reg_gen_dec, nets_to_train, retain_graph=True)
                                reg_gen_dec_sum += reg_gen_dec.item() / g_reg_every

                            loss_gen_dec = (1 / batch_mult) * 0.5 * compute_gan_loss(labs_dec, 1)
                            model_manager.loss_backward(loss_gen_dec, nets_to_train, retain_graph=True)
                            # loss_gen_dec_inv = (1 / batch_mult) * 0.5 * -compute_gan_loss(labs_dec, 0)
                            # model_manager.loss_backward(loss_gen_dec_inv, nets_to_train, retain_graph=True)
                            loss_gen_dec_sum += loss_gen_dec.item()

                            corrupt_inits = rand_circle_masks(out_embs[-1])
                            images_redec, _, _ = decoder(lat_gen, corrupt_inits[model_manager.it % 3])

                            lat_top_redec, _, _ = dis_encoder(images_redec, labels)
                            labs_redec = discriminator(lat_top_redec)

                            loss_gen_redec = (1 / batch_mult) * 0.5 * compute_gan_loss(labs_redec, 1)
                            model_manager.loss_backward(loss_gen_redec, nets_to_train)
                            # loss_gen_redec_inv = (1 / batch_mult) * 0.5 * -compute_gan_loss(labs_redec, 0)
                            # model_manager.loss_backward(loss_gen_redec_inv, nets_to_train, retain_graph=True)
                            loss_gen_redec_sum += loss_gen_redec.item()

                        # if isinstance(generator, torch.nn.DataParallel):
                        #     grad_ema_update(generator.module)
                        # else:
                        #     grad_ema_update(generator)

                        # grad_mult(decoder, (0.5 * (g_factor_enc + g_factor_dec)) ** 0.5)
                        # grad_mult(generator, (0.5 * (g_factor_enc + g_factor_dec)) ** 0.5)

                    return loss_gen_dec_sum, loss_gen_redec_sum, reg_gen_dec_sum, trainiter, pl_mean_dec

                train_order = [True, False]
                train_order = np.random.permutation(train_order)
                for turn in train_order:
                    if turn:
                        (loss_dis_enc_sum, labs_dis_enc_sign, labs_dis_enc_sign_mm, reg_dis_enc_sum, new_weights,
                         loss_dis_dec_sum, labs_dis_dec_sign, labs_dis_dec_sign_mm, reg_dis_dec_sum, g_factor_enc, g_factor_dec,
                         loss_dis_redec_sum, trainiter, d_reg_param_mean, d_reg_every_mean, d_reg_every_mean_next) = train_d(loss_dis_enc_sum, labs_dis_enc_sign, labs_dis_enc_sign_mm, reg_dis_enc_sum, new_weights,
                                                                                                                             loss_dis_dec_sum, labs_dis_dec_sign, labs_dis_dec_sign_mm, reg_dis_dec_sum, g_factor_enc, g_factor_dec,
                                                                                                                             loss_dis_redec_sum, trainiter, d_reg_param_mean, d_reg_every_mean, d_reg_every_mean_next)
                    else:
                        loss_gen_dec_sum, loss_gen_redec_sum, reg_gen_dec_sum, trainiter, pl_mean_dec = train_g(loss_gen_dec_sum, loss_gen_redec_sum, reg_gen_dec_sum, trainiter, pl_mean_dec)

                # Streaming Images
                with torch.no_grad():
                    images_gen_l = []
                    for i in range(batch_split):
                        ini = i * batch_split_size
                        end = (i + 1) * batch_split_size
                        lat_gen = generator(z_test[ini:end], labels_test[ini:end])
                        images_gen, out_embs, _ = decoder(lat_gen)
                        out_embs[-1] = rand_circle_masks(out_embs[-1], images_test_mask[ini:end])[0]
                        images_regen, _, _ = decoder(lat_gen, out_embs[-1])
                        images_gen = torch.cat([images_gen, images_regen], dim=3)
                        images_gen_l.append(images_gen)
                    images_gen = torch.cat(images_gen_l)

                stream_images(images_gen, config_name + '/gax', config['training']['out_dir'] + '/gax')

                # Print progress
                running_loss_dis[batch % window_size] = loss_dis_enc_sum + loss_dis_dec_sum
                running_loss_gen[batch % window_size] = loss_gen_dec_sum
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
                model_manager.log_scalar('losses',  'loss_dis_redec',  loss_dis_redec_sum)
                model_manager.log_scalar('losses',  'loss_gen_dec',  loss_gen_dec_sum)
                model_manager.log_scalar('losses',  'loss_gen_redec',  loss_gen_redec_sum)

                # model_manager.log_scalar('regs',  'g_factor_enc',  g_factor_enc)
                # model_manager.log_scalar('regs',  'g_factor_dec',  g_factor_dec)
                model_manager.log_scalar('regs',  'reg_dis_enc',  reg_dis_enc_sum)
                model_manager.log_scalar('regs',  'reg_dis_dec',  reg_dis_dec_sum)
                model_manager.log_scalar('regs',  'labs_dis_enc_sign_mm',  labs_dis_enc_sign_mm)
                model_manager.log_scalar('regs',  'labs_dis_dec_sign_mm',  labs_dis_dec_sign_mm)
                # model_manager.log_scalar('regs',  'reg_dis_unb',  reg_dis_unb_sum)
                model_manager.log_scalar('regs',  'd_reg_every_mean',  d_reg_every_mean)
                model_manager.log_scalar('regs',  'd_reg_param_mean',  d_reg_param_mean)
                # model_manager.log_scalar('regs',  'd_unb_param_mean',  d_unb_param_mean)

                if g_reg_every > 0:
                    model_manager.log_scalar('regs',  'reg_gen_dec',  reg_gen_dec_sum)

                    model_manager.log_scalar('regs',  'pl_mean_dec',  pl_mean_dec)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((model_manager.epoch + 1) % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')
            images, labels, _, trainiter, _ = get_inputs(trainiter, batch_size, device)
            images_gen_l = []
            for i in range(batch_split):
                ini = i * batch_split_size
                end = (i + 1) * batch_split_size
                lat_gen = generator(z_test[ini:end], labels_test[ini:end])
                images_gen, out_embs, _ = decoder(lat_gen)
                out_embs[-1] = rand_circle_masks(out_embs[-1], images_test_mask[ini:end])[0]
                images_regen, _, _ = decoder(lat_gen, out_embs[-1])
                images_gen = torch.cat([images_gen, images_regen], dim=3)
                images_gen_l.append(images_gen)
            images_gen = torch.cat(images_gen_l)
            model_manager.log_images(images,  'all_input')
            model_manager.log_images(images_gen,  'all_gen')
            for lab in range(config['training']['sample_labels']):
                if labels_test.dim() == 1:
                    fixed_lab = torch.full((batch_size,), lab, device=device, dtype=torch.int64)
                else:
                    fixed_lab = labels_test.clone()
                    fixed_lab[:, lab] = 1
                images_gen_l = []
                for i in range(batch_split):
                    ini = i * batch_split_size
                    end = (i + 1) * batch_split_size
                    lat_gen = generator(z_test[ini:end], fixed_lab[ini:end])
                    images_gen, out_embs, _ = decoder(lat_gen)
                    out_embs[-1] = rand_circle_masks(out_embs[-1], images_test_mask[ini:end])[0]
                    images_regen, _, _ = decoder(lat_gen, out_embs[-1])
                    images_gen = torch.cat([images_gen, images_regen], dim=3)
                    images_gen_l.append(images_gen)
                images_gen = torch.cat(images_gen_l)
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

