# -*- coding: utf-8 -*-
import math
import numpy as np
import torch
import torch.nn.functional as F
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import rand_erase_images, rand_change_letters, rand_circle_masks
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
from os.path import basename, splitext

torch.backends.cudnn.deterministic = False
torch.backends.cudnn.benchmark = True

np.random.seed(42)
torch.manual_seed(42)

parser = argparse.ArgumentParser(description='Train a MultiGoal-CAE')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = splitext(basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
n_epochs = config['training']['n_epochs']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
batch_mult_steps = config['training']['batch_mult_steps']
n_workers = config['training']['n_workers']
n_goals = config['network']['kwargs']['n_goals'] if 'n_goals' in config['network']['kwargs'] else 2
config['network']['kwargs']['n_goals'] = n_goals
last_ret = config['network']['kwargs']['last_ret'] if 'last_ret' in config['network']['kwargs'] else True
config['network']['kwargs']['last_ret'] = last_ret



# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_split_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

config['training']['batches_per_epoch'] = len(trainloader) // batch_split

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'Encoder'},
    'lat_compressor': {'class': 'base', 'sub_class': 'LatCompressor'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}

model_manager = ModelManager('mgcae', networks_dict, config)
encoder = model_manager.get_network('encoder')
lat_compressor = model_manager.get_network('lat_compressor')
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
    images = (images + 1. / 128 * torch.randn_like(images)).clamp_(-1.0, 1.0)
    images, labels = images.to(device), labels.to(device)
    return images, labels, trainiter


goals_test = []
trainiter = iter(trainloader)
for g in range(n_goals):
    images_test, labels_test, trainiter = get_inputs(trainiter, batch_size, device)
    goals_test.append(images_test)

window_size = math.ceil((len(trainloader) // batch_split) / 10)

for _ in range(model_manager.epoch, n_epochs):
    with model_manager.on_epoch():

        running_loss = np.zeros(window_size)

        batch_mult = (int((model_manager.epoch / n_epochs) * batch_mult_steps) + 1) * batch_split

        t = trange(config['training']['batches_per_epoch'] - (model_manager.it % config['training']['batches_per_epoch']))
        t.set_description('| ep: %d | lr: %.2e |' % (model_manager.epoch, model_manager.lr))
        for batch in t:

            with model_manager.on_batch():

                loss_dec_sum = 0

                with model_manager.on_step(['encoder', 'decoder', 'lat_compressor']) as nets_to_train:

                    for _ in range(batch_mult):

                        goals = []
                        for g in range(n_goals):
                            images, _, trainiter = get_inputs(trainiter, batch_split_size, device)
                            goals.append(images)

                        lats = []
                        for g in range(n_goals):
                            lat_enc, _, _ = encoder(goals[g])
                            lats.append(lat_enc)

                        lat_enc = lat_compressor(lats)

                        init_samples = None
                        for g in range(n_goals):
                            _, out_embs, images_redec_raw = decoder(lat_enc, init_samples)
                            init_samples = out_embs[-1]

                            loss_dec = (1 / batch_mult) * F.mse_loss(images_redec_raw, goals[g])
                            model_manager.loss_backward(loss_dec, nets_to_train, retain_graph=True)
                            loss_dec_sum += loss_dec.item()

                            if last_ret and g == 0:
                                first_init = init_samples

                        if last_ret:
                            _, out_embs, _ = decoder(lat_enc, init_samples)

                            loss_dec = (1 / batch_mult) * F.mse_loss(out_embs[-1], first_init)
                            model_manager.loss_backward(loss_dec, nets_to_train)
                            loss_dec_sum += loss_dec.item()

                # Streaming Images
                with torch.no_grad():
                    lats = []
                    for g in range(n_goals):
                        lat_enc, _, _ = encoder(goals_test[g])
                        lats.append(lat_enc)

                    lat_enc = lat_compressor(lats)

                    init_samples = None
                    images_dec_l = []
                    for g in range(n_goals * (3 if last_ret else 1)):
                        images_dec, out_embs, _ = decoder(lat_enc, init_samples)
                        init_samples = out_embs[-1]
                        images_dec_l.append(images_dec)
                    images_dec = torch.cat(images_dec_l, dim=3)

                stream_images(images_dec, config_name, config['training']['out_dir'])

                # Print progress
                running_loss[batch % window_size] = loss_dec_sum
                running_factor = window_size if batch > window_size else batch + 1
                t.set_postfix(loss='%.2e' % (np.sum(running_loss) / running_factor))

                # Log progress
                model_manager.log_scalar('learning_rates',  'all',  model_manager.lr)
                if model_manager.momentum is not None:
                    model_manager.log_scalar('learning_rates',  'all_mom',  model_manager.momentum)

                model_manager.log_scalar('losses',  'loss_dec',  loss_dec_sum)

    with torch.no_grad():
        # Log images
        if config['training']['sample_every'] > 0 and ((model_manager.epoch + 1) % config['training']['sample_every']) == 0:
            model_manager.save()
            t.write('Creating samples...')

            goals = []
            for g in range(n_goals):
                images, _, trainiter = get_inputs(trainiter, batch_size, device)
                goals.append(images)

            lats = []
            for g in range(n_goals):
                lat_enc, _, _ = encoder(goals[g])
                lats.append(lat_enc)

            lat_enc = lat_compressor(lats)

            init_samples = None
            images_dec_l = []
            for g in range(n_goals * (3 if last_ret else 1)):
                images_dec, out_embs, _ = decoder(lat_enc, init_samples)
                init_samples = out_embs[-1]
                images_dec_l.append(images_dec)

            images = torch.cat(goals, dim=3)
            images_dec = torch.cat(images_dec_l, dim=3)

            model_manager.log_images(images,  'all_input')
            model_manager.log_images(images_dec,  'all_dec')

model_manager.save()
print('Training is complete...')
