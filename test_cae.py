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
from src.model_manager import ModelManager
import os
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Train a FractalNet')
parser.add_argument('config', type=str, help='Path to config file.')
args = parser.parse_args()
config = load_config(args.config)
config_name = os.path.splitext(os.path.basename(args.config))[0]

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
torch.multiprocessing.set_sharing_strategy('file_system')

image_size = config['data']['image_size']
channels = config['data']['channels']
n_filter = config['network']['kwargs']['n_filter']
letter_encoding = config['network']['kwargs']['letter_encoding']
persistence = config['training']['persistence']
regeneration = config['training']['regeneration']
batch_size = config['training']['batch_size']
batch_split = config['training']['batch_split']
batch_split_size = batch_size // batch_split
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

model_manager = ModelManager('cae', networks_dict, config, False)
encoder = model_manager.get_network('encoder')
decoder = model_manager.get_network('decoder')
if letter_encoding:
    letter_encoder = model_manager.get_network('letter_encoder')
    letter_decoder = model_manager.get_network('letter_decoder')

model_manager.print()


def enc(images):
    lat_enc, _, _ = encoder(images)
    if letter_encoding:
        lat_enc = letter_encoder(lat_enc)
    return lat_enc


def dec(lat_enc, init_seed=None):
    if letter_encoding:
        lat_dec = letter_decoder(lat_enc)
    else:
        lat_dec = lat_enc
    images_dec, out_embs, _ = decoder(lat_dec, init_seed)
    return images_dec, out_embs


def forward_pass(images, init_seed=None):
    lat_enc = enc(images)
    images_dec, out_embs = dec(lat_enc, init_seed)
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


def save_imgs(images, out_dir, tag, make_video=False, nrow=8):
    if not os.path.exists(out_dir):
        os.makedirs(out_dir)
    images_file = os.path.join(out_dir, '%s.png' % tag)

    images = (images * 0.5) + 0.5
    torchvision.utils.save_image(images, images_file, nrow=nrow)  # imgs.size(0))

    if make_video:
        if images.size(1) == 4:
            alpha = images[:, 3:4, :, :].clamp(0.0, 1.0)
            images = images[:, :3, :, :] * alpha
            images = (1.0 - alpha) + images

        images = images.clamp(0.0, 1.0)
        images = images * 255

        h, w = images.size()[2:]
        video_file = os.path.join(out_dir, '%s.mp4' % tag)
        video_writer = FFMPEG_VideoWriter(size=(w, h), filename=video_file, fps=images.size(0) / 8)
        for i in range(images.size(0)):
            video_writer.write_frame(np.uint8(images[i, ...].permute(1, 2, 0).cpu().numpy()))
        for _ in range(images.size(0) // 4):
            video_writer.write_frame(np.uint8(images[-1, ...].permute(1, 2, 0).cpu().numpy()))
        video_writer.close()


transform = torchvision.transforms.Compose([
    torchvision.transforms.Resize(image_size),
    torchvision.transforms.ToTensor(),
    torchvision.transforms.Lambda(lambda x: (x - 0.5) * 2.0),
    torchvision.transforms.Lambda(lambda x: x.reshape(1, channels, image_size, image_size)),
])


def load_image(img_loc):
    image = Image.open(img_loc)
    if channels == 3:
        image = image.convert("RGB")
    elif channels == 4:
        image = image.convert("RGBA")
    image = transform(image)
    image = image.to(device)
    return image



normal_emos_ids = ["emoji_u1f610", "emoji_u1f615", "emoji_u1f617", "emoji_u1f626",
                   "emoji_u1f636", "emoji_u1f641", "emoji_u1f642", "emoji_u2639" ]

smily_emos_ids = ["emoji_u1f600", "emoji_u1f602", "emoji_u1f603", "emoji_u1f604",
                  "emoji_u1f605", "emoji_u1f606", "emoji_u1f913", "emoji_u1f929"]

tongue_emos_ids = ["emoji_u1f61b", "emoji_u1f61c", "emoji_u1f61d"]

heartface_emos_ids = ["emoji_u1f60d", "emoji_u1f63b"]

other_emos_ids = ["emoji_u1f60e", "emoji_u1f611", "emoji_u1f911"]


with torch.no_grad():
    print('Performing PCA...')
    t = trange((len(trainset) // batch_size) + 1)
    lat_encs = []
    lat_labels = []
    for batch in t:
        idxs = torch.arange(batch * batch_size, min((batch + 1) * batch_size, len(trainset)))
        images, labels = get_inputs(idxs, device)
        lat_labels.append(np.array(labels))
        lat_enc = enc(images)
        if letter_encoding:
            lat_enc = lat_enc.reshape(lat_enc.size(0), -1)
        lat_encs.append(lat_enc)
    lat_labels = np.concatenate(lat_labels, 0)
    lat_encs = torch.cat(lat_encs, 0).cpu().numpy()
    pca = PCA(n_components=2)
    lat_enc_pca = pca.fit_transform(lat_encs)

    label_names = [i for i in range(config['data']['n_labels'])]
    colors = matplotlib.colors.TABLEAU_COLORS

    plt.figure(figsize=(8, 8))
    for color, i, target_name in zip(colors, [i for i in range(config['data']['n_labels'])], label_names):
        plt.scatter(lat_enc_pca[lat_labels == i, 0], lat_enc_pca[lat_labels == i, 1],
                    color=color, lw=2, label=target_name)

    plt.title("title")
    plt.legend(loc="best", shadow=False, scatterpoints=1)
    # plt.axis([-4, 4, -1.5, 1.5])

    # plt.show()
    pca_dir = os.path.join(config['training']['out_dir'], 'test', 'pca')
    if not os.path.exists(pca_dir):
        os.makedirs(pca_dir)
    plt.savefig(os.path.join(pca_dir, 'plot.png'))

    normal_emos = torch.cat([load_image(os.path.join(config['data']['train_dir'], 'smileys_and_emotion', '%s.png' % id)) for id in normal_emos_ids])
    normal_enc = enc(normal_emos)
    smily_emos = torch.cat([load_image(os.path.join(config['data']['train_dir'], 'smileys_and_emotion', '%s.png' % id)) for id in smily_emos_ids])
    smily_enc = enc(smily_emos)
    tongue_emos = torch.cat([load_image(os.path.join(config['data']['train_dir'], 'smileys_and_emotion', '%s.png' % id)) for id in tongue_emos_ids])
    tongue_enc = enc(tongue_emos)
    other_emos = torch.cat([load_image(os.path.join(config['data']['train_dir'], 'smileys_and_emotion', '%s.png' % id)) for id in other_emos_ids])
    other_enc = enc(other_emos)

    normal_enc_common = (normal_enc.mean(dim=0, keepdim=True) > 0.5).to(torch.float32)
    normal_enc_common_mask = 1. - normal_enc_common.sum(dim=1, keepdim=True).clamp_max_(1.0)
    lat_dec = normal_enc_common + normal_enc_common_mask * other_enc[1:2, ...]

    images_dec = dec(lat_dec)
    print(images_dec.size())
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'pca'), 'dec', nrow=4)


    print('Plotting Randdom CAs...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)
    save_imgs(images, os.path.join(config['training']['out_dir'], 'test', 'random'), 'input')

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'random'), 'dec')
    for i in range(batch_size):
        save_imgs(out_embs[1:, i, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test', 'random'), '%d' % i, True)

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
    print('MSE: ', losses.mean().item())

    l_sort = torch.argsort(losses)
    sorted_idxs = torch.arange(len(trainset))[l_sort]

    print('Plotting Best CAs...')
    images, labels = get_inputs(sorted_idxs[:batch_size], device)
    save_imgs(images, os.path.join(config['training']['out_dir'], 'test', 'best'), 'input')

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'best'), 'dec')
    for i in range(batch_size):
        save_imgs(out_embs[1:, i, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test', 'best'), '%d' % i, True)

    print('Plotting Worst CAs...')
    images, labels = get_inputs(sorted_idxs[-batch_size:], device)
    save_imgs(images, os.path.join(config['training']['out_dir'], 'test', 'worst'), 'input')

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'worst'), 'dec')
    for i in range(batch_size):
        save_imgs(out_embs[1:, i, :config['data']['channels'], :, :], os.path.join(config['training']['out_dir'], 'test', 'worst'), '%d' % i, True)

    print('Plotting Persistence...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)

    for n in range(1, 19):
        images_dec, out_embs = forward_pass(images, None if n == 1 else out_embs[-1, ...])
        save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'persistence'), '%d' % n)

    print('Plotting Regeneration...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)
    _, out_embs = forward_pass(images)
    init_image = out_embs[-1, ...]

    init_occ = init_image.clone()
    init_occ[:, :, image_size // 2:, :] = 0.0
    save_imgs(init_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '0_occ')
    for _ in range(4):
        images_dec, out_embs = forward_pass(images, init_occ)
        init_occ = out_embs[-1, ...]
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '0')

    init_occ = init_image.clone()
    init_occ[:, :, :image_size // 2, :] = 0.0
    save_imgs(init_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '1_occ')
    for _ in range(4):
        images_dec, out_embs = forward_pass(images, init_occ)
        init_occ = out_embs[-1, ...]
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '1')

    init_occ = init_image.clone()
    init_occ[:, :, :, image_size // 2:] = 0.0
    save_imgs(init_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '2_occ')
    for _ in range(4):
        images_dec, out_embs = forward_pass(images, init_occ)
        init_occ = out_embs[-1, ...]
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '2')

    init_occ = init_image.clone()
    init_occ[:, :, :, :image_size // 2] = 0.0
    save_imgs(init_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '3_occ')
    for _ in range(4):
        images_dec, out_embs = forward_pass(images, init_occ)
        init_occ = out_embs[-1, ...]
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '3')

    init_occ = init_image.clone()
    init_occ[:, :, image_size // 4:(image_size // 4) * 3, image_size // 4:(image_size // 4) * 3] = 0.0
    save_imgs(init_occ[:, :config['data']['channels'], ...], os.path.join(config['training']['out_dir'], 'test', 'regen'), '4_occ')
    for _ in range(4):
        images_dec, out_embs = forward_pass(images, init_occ)
        init_occ = out_embs[-1, ...]
    save_imgs(images_dec, os.path.join(config['training']['out_dir'], 'test', 'regen'), '4')

