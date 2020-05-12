# -*- coding: utf-8 -*-
import time
import numpy as np
import torch
import torch.nn.functional as F
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
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter

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


def forward_pass(images, init_seed=None):
    lat_enc, _, _ = encoder(images)
    if letter_encoding:
        letters = letter_encoder(lat_enc)
        lat_dec = letter_decoder(letters)
    else:
        lat_dec = lat_enc
    images_dec, out_embs, _ = decoder(lat_dec, init_seed)
    out_embs = torch.stack(out_embs, 0)
    return images_dec, out_embs


def get_inputs(idxs, device):
    images = []
    labels = []
    for idx in idxs:
        image, label = trainset[idx]
        images.append(image)
        labels.append(label)
    images = torch.stack(images, 0).to(device)
    return images, labels


def save_imgs(images, out_dir, tag, make_video=False):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    images_file = os.path.join(out_dir, '%s.png' % tag)

    images = (images * 0.5) + 0.5
    torchvision.utils.save_image(images, images_file, nrow=8)  # imgs.size(0))

    if make_video:
        if images.size(1) == 4:
            alpha = images[:, 3:4, :, :].clamp(0.0, 1.0)
            images = images[:, :3, :, :] * alpha
            images = ((1.0 - alpha) + images).clamp(0.0, 1.0)

        images = images * 255

        h, w = images.size()[2:]
        video_file = os.path.join(out_dir, '%s.mp4' % tag)
        video_writer = FFMPEG_VideoWriter(size=(w, h), filename=video_file, fps=images.size(0) / 8)
        for i in range(images.size(0)):
            video_writer.write_frame(np.uint8(images[i, ...].permute(1, 2, 0).cpu().numpy()))
        for _ in range(images.size(0) // 4):
            video_writer.write_frame(np.uint8(images[-1, ...].permute(1, 2, 0).cpu().numpy()))
        video_writer.close()


with torch.no_grad():
    print('Computing Error...')
    t = trange((len(trainset) // batch_size) + 1)
    losses = []
    for batch in t:
        idxs = torch.arange(batch * batch_size, min((batch + 1) * batch_size, len(trainset)))
        images, labels = get_inputs(idxs, device)
        images_dec, out_embs = forward_pass(images)
        loss = F.mse_loss(images_dec, images, reduction='none').mean(dim=(1, 2, 3))
        losses.append(loss)
    losses = torch.cat(losses, 0)
    print('MSE: ', losses.mean())

    l_sort = torch.argsort(losses)
    sorted_idxs = torch.arange(len(trainset))[l_sort]

    print('Plotting Best CAs...')
    images, labels = get_inputs(sorted_idxs[:batch_size], device)

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'best'), 'all')
    for i in range(batch_size):
        save_imgs(out_embs[:, i, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test', 'best'), '%d' % i, True)

    print('Plotting Worst CAs...')
    images, labels = get_inputs(sorted_idxs[-batch_size:], device)

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'worst'), 'all')
    for i in range(batch_size):
        save_imgs(out_embs[:, i, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test', 'worst'), '%d' % i, True)

    print('Plotting Persistence...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)

    for n in range(1, 9):
        images_dec, out_embs = forward_pass(images, None if n == 1 else out_embs[-1, ...])
        save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'persist'), '%d' % n)

    print('Plotting Regeneration...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)

    images_occ = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    images_occ[:, :config['data']['channels'], ...] = images.clone()
    images_occ[:, :, image_size // 2:, :] = 0.0
    images_dec, out_embs = forward_pass(images, images_occ)
    save_imgs(images_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '0_occ')
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '0')

    images_occ = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    images_occ[:, :config['data']['channels'], ...] = images.clone()
    images_occ[:, :, :image_size // 2, :] = 0.0
    images_dec, out_embs = forward_pass(images, images_occ)
    save_imgs(images_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '1_occ')
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '1')

    images_occ = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    images_occ[:, :config['data']['channels'], ...] = images.clone()
    images_occ[:, :, :, image_size // 2:] = 0.0
    images_dec, out_embs = forward_pass(images, images_occ)
    save_imgs(images_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '2_occ')
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '2')

    images_occ = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    images_occ[:, :config['data']['channels'], ...] = images.clone()
    images_occ[:, :, :, :image_size // 2] = 0.0
    images_dec, out_embs = forward_pass(images, images_occ)
    save_imgs(images_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '3_occ')
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '3')

    images_occ = torch.zeros(batch_size, n_filter, image_size, image_size, device=device)
    images_occ[:, :config['data']['channels'], ...] = images.clone()
    images_occ[:, :, image_size // 4:(image_size // 4) * 3, image_size // 4:(image_size // 4) * 3] = 0.0
    images_dec, out_embs = forward_pass(images, images_occ)
    save_imgs(images_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '4_occ')
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '4')

