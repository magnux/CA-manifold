# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torchvision
import argparse
from tqdm import trange
from src.config import load_config
from src.inputs import get_dataset
from src.utils.media_utils import save_images, rand_erase_images, rand_change_letters, rand_circle_masks
from src.utils.model_utils import ca_seed, SamplePool
from src.model_manager import ModelManager
from src.utils.web.webstreaming import stream_images
import os

parser = argparse.ArgumentParser(description='Train a FractalNet')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = os.path.splitext(os.path.basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
n_filter = config['network']['kwargs']['n_filter']
letter_encoding = config['network']['kwargs']['letter_encoding']
use_sample_pool = config['training']['sample_pool']
damage_init = config['training']['damage_init']
batch_size = config['training']['batch_size']
n_workers = config['training']['n_workers']

# Inputs
trainset = get_dataset(name=config['data']['name'], type=config['data']['type'],
                       data_dir=config['data']['train_dir'], size=config['data']['image_size'])
trainloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size,
                                          shuffle=True, num_workers=n_workers, drop_last=True)

# Networks
networks_dict = {
    'encoder': {'class': config['network']['class'], 'sub_class': 'Encoder'},
    'decoder': {'class': config['network']['class'], 'sub_class': 'Decoder'},
}
if letter_encoding:
    networks_dict.update({
        'letter_encoder': {'class': 'base', 'sub_class': 'LetterEncoder'},
        'letter_decoder': {'class': 'base', 'sub_class': 'LetterDecoder'},
    })

model_manager = ModelManager('cae', networks_dict, config)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
if letter_encoding:
    letter_encoder = model_manager.get_network('letter_encoder')
    letter_decoder = model_manager.get_network('letter_decoder')

model_manager.print()


def get_inputs(trainiter, batch_size, device):
    next_inputs = next(trainiter, None)
    if trainiter is None or next_inputs is None:
        trainiter = iter(trainloader)
        next_inputs = next(trainiter, None)
    images, labels = next_inputs
    images, labels = images[:batch_size, ...], labels[:batch_size, ...]
    images, labels = images.to(device), labels.to(device)
    return images, labels, trainiter


images_test, labels_test, trainiter = get_inputs(iter(trainloader), batch_size, device)

if use_sample_pool:
    n_slots = len(trainset) * 16
    target = []
    for _ in range((n_slots // batch_size) + 1):
        images, _, trainiter = get_inputs(trainiter, batch_size, torch.device('cpu'))
        target.append(images)
    target = torch.cat(target, dim=0)[:n_slots, ...].numpy()
    seed = ca_seed(n_slots, n_filter, image_size, torch.device('cpu')).numpy()
    sample_pool = SamplePool(target=target, init=seed)
    frac_size = batch_size // 16
    frac_seed = ca_seed(frac_size, n_filter, image_size, torch.device('cpu')).numpy()


def save_imgs(imgs, outdir, tag):
    if not os.path.exists(outdir):
        os.makedirs(outdir)
    outfile = os.path.join(outdir, '%s.png' % tag)

    imgs = (imgs * 0.5) + 0.5
    torchvision.utils.save_image(imgs, outfile, nrow=8)  # imgs.size(0))


its = [i for i in range(64)]
# post_its = [2 ** i for i in range(6, 11)]
idxs = [1, 3, 5, 7]
with torch.no_grad():
    model_manager.set_n_calls('decoder', 64)
    lat_enc, _, _ = encoder(images_test)
    if letter_encoding:
        letters = letter_encoder(lat_enc)
        lat_dec = letter_decoder(letters)
    else:
        lat_dec = lat_enc
    images_dec, out_embs, _ = decoder(lat_dec)
    out_embs = torch.stack(out_embs, 0)
    for idx in idxs:
        save_imgs(out_embs[its, idx, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test'), '%d_its' % idx)
        # save_imgs(out_embs[its, idx, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test'), '%d_post_its' % idx)
