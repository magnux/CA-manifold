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
import shutil
from moviepy.video.io.ffmpeg_writer import FFMPEG_VideoWriter
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import matplotlib
from PIL import Image

np.random.seed(42)
torch.manual_seed(42)
torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False

parser = argparse.ArgumentParser(description='Test a CAE')
parser.add_argument('config', type=str, help='Path to config file.')
parser.add_argument('-clear', action="store_true", help='Clear previous tests')
parser.add_argument('-mse', action="store_true", help='Mean squared error')
parser.add_argument('-persist', action="store_true", help='Persistence tests')
parser.add_argument('-regen', action="store_true", help='Regeneration tests')
parser.add_argument('-pca', action="store_true", help='PCA of encoding')
parser.add_argument('-arithmetic', action="store_true", help='Encoding arithmetic tests')
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
batch_size = 16
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
    out_embs = torch.stack(out_embs, 0)
    return images_dec, out_embs


def forward_pass(images, init_seed=None):
    lat_enc = enc(images)
    images_dec, out_embs = dec(lat_enc, init_seed)
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


def save_imgs(images, out_dir, tag, make_video=False, nrow=batch_size):
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

smiley_emos_ids = ["emoji_u1f600", "emoji_u1f602", "emoji_u1f603", "emoji_u1f604",
                   "emoji_u1f605", "emoji_u1f606", "emoji_u1f913", "emoji_u1f929"]

tongue_emos_ids = ["emoji_u1f61b", "emoji_u1f61c", "emoji_u1f61d"]

heartface_emos_ids = ["emoji_u1f60d", "emoji_u1f63b"]

other_emos_ids = ["emoji_u1f60e", "emoji_u1f611", "emoji_u1f911"]

super_girl_id = "emoji_u1f9b8_1f3ff_200d_2640"

unicorn_id = "emoji_u1f984"

test_dir = os.path.join(config['training']['out_dir'], 'test')
if args.clear:
    if os.path.exists(test_dir):
        shutil.rmtree(test_dir)

with torch.no_grad():
    print('Plotting Randdom CAs...')
    images, labels = get_inputs(np.random.choice(len(trainset), batch_size, False), device)
    save_imgs(images, os.path.join(test_dir, 'random'), 'input')

    images_dec, out_embs = forward_pass(images)
    save_imgs(images_dec, os.path.join(test_dir, 'random'), 'dec_%s' % config_name)
    for i in range(batch_size):
        save_imgs(out_embs[1:, i, :config['data']['channels'], :, :], os.path.join(test_dir, 'random'), '%d' % i, True)

    if args.mse:
        print('Computing MSE...')
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
        save_imgs(images, os.path.join(test_dir, 'best'), 'input')

        images_dec, out_embs = forward_pass(images)
        save_imgs(images_dec, os.path.join(test_dir, 'best'), 'dec_%s' % config_name)
        for i in range(batch_size):
            save_imgs(out_embs[1:, i, :channels, :, :], os.path.join(test_dir, 'best'), '%d' % i, True)

        print('Plotting Worst CAs...')
        images, labels = get_inputs(sorted_idxs[-batch_size:], device)
        save_imgs(images, os.path.join(test_dir, 'worst'), 'input')

        images_dec, out_embs = forward_pass(images)
        save_imgs(images_dec, os.path.join(test_dir, 'worst'), 'dec_%s' % config_name)
        for i in range(batch_size):
            save_imgs(out_embs[1:, i, :channels, :, :], os.path.join(test_dir, 'worst'), '%d' % i, True)

    if args.persist:
        print('Plotting Persistence...')
        unicorn = load_image(os.path.join(config['data']['train_dir'], 'animals_and_nature', '%s.png' % unicorn_id))
        unicorn_enc = enc(unicorn)

        num_passes = 8
        images_out = []
        for n in range(1, (2 ** num_passes) + 1):
            images_dec, out_embs = dec(unicorn_enc, None if n == 1 else out_embs[-1, ...])
            if n == 1:
                images_out.append(out_embs[0, :, :channels, ...])
            if np.log2(n) % 1. == 0:
                images_out.append(images_dec)
        images_out = torch.cat(images_out, 0)
        save_imgs(images_out, os.path.join(test_dir, 'persistence'), 'dec_%s' % config_name, nrow=10)

    if args.regen:
        print('Plotting Regeneration...')
        super_girl = load_image(os.path.join(config['data']['train_dir'], 'people_and_body', '%s.png' % super_girl_id))
        super_girl_enc = enc(super_girl)
        _, out_embs = dec(super_girl_enc)
        init_image = out_embs[-1, ...]

        def regen_test(images_enc, init_image, occ_mask, save_prefix):
            init_occ = init_image.clone()
            init_occ *= occ_mask
            num_passes = 7
            images_out = [init_image[:, :channels, ...], init_occ[:, :channels, ...]]
            for n in range(1, (2 ** num_passes) + 1):
                images_dec, out_embs = dec(images_enc, init_occ)
                init_occ = out_embs[-1, ...]
                if np.log2(n) % 1. == 0:
                    images_out.append(images_dec)
            images_out = torch.cat(images_out, 0)
            save_imgs(images_out, os.path.join(test_dir, 'regen'), save_prefix)

        occ_mask = torch.ones_like(init_image)
        occ_mask[:, :, image_size // 2:, :] = 0.0
        regen_test(super_girl_enc, init_image, occ_mask, '0_%s' % config_name)

        occ_mask = torch.ones_like(init_image)
        occ_mask[:, :, :image_size // 2, :] = 0.0
        regen_test(super_girl_enc, init_image, occ_mask, '1_%s' % config_name)

        occ_mask = torch.ones_like(init_image)
        occ_mask[:, :, :, image_size // 2:] = 0.0
        regen_test(super_girl_enc, init_image, occ_mask, '2_%s' % config_name)

        occ_mask = torch.ones_like(init_image)
        occ_mask[:, :, :, :image_size // 2] = 0.0
        regen_test(super_girl_enc, init_image, occ_mask, '3_%s' % config_name)

        occ_mask = torch.ones_like(init_image)
        occ_mask[:, :, image_size // 4:(image_size // 4) * 3, image_size // 4:(image_size // 4) * 3] = 0.0
        regen_test(super_girl_enc, init_image, occ_mask, '4_%s' % config_name)

    if args.pca:
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
        pca_dir = os.path.join(test_dir, 'pca')
        if not os.path.exists(pca_dir):
            os.makedirs(pca_dir)
        plt.savefig(os.path.join(pca_dir, 'plot.png'))

    if args.arithmetic:
        print('Performing arithmetic tests...')
        smiley_dir = os.path.join(config['data']['train_dir'], 'smileys_and_emotion')
        normal_emos = torch.cat([load_image(os.path.join(smiley_dir, '%s.png' % id)) for id in normal_emos_ids])
        normal_enc = enc(normal_emos)
        smiley_emos = torch.cat([load_image(os.path.join(smiley_dir, '%s.png' % id)) for id in smiley_emos_ids])
        smiley_enc = enc(smiley_emos)
        tongue_emos = torch.cat([load_image(os.path.join(smiley_dir, '%s.png' % id)) for id in tongue_emos_ids])
        tongue_enc = enc(tongue_emos)
        other_emos = torch.cat([load_image(os.path.join(smiley_dir, '%s.png' % id)) for id in other_emos_ids])
        other_enc = enc(other_emos)
        
        ari_dir = os.path.join(test_dir, 'lat_ari')
        
        save_imgs(normal_emos, ari_dir, 'ari_normal', nrow=len(normal_emos_ids) // 2)
        save_imgs(smiley_emos, ari_dir, 'ari_smiley', nrow=len(smiley_emos_ids) // 2)
        save_imgs(other_emos, ari_dir, 'ari_other', nrow=len(other_emos_ids))

        normal_enc_common = torch.softmax(normal_enc.mean(dim=0, keepdim=True) * 100, dim=1)
        normal_common_dec, _ = dec(normal_enc_common)
        save_imgs(normal_common_dec, ari_dir, 'ari_normal_common_dec_%s' % config_name, nrow=1)
        inj_mask = (torch.rand((1, 1, normal_enc_common.size(2)), device=device) > 0.35).to(torch.float32)
        lat_dec = other_enc.clone()
        for i in range(len(other_emos_ids)):
            lat_dec[i:i + 1, ...] = inj_mask * normal_enc_common + (1 - inj_mask) * lat_dec[i:i + 1, ...]

        images_dec, _ = dec(lat_dec)
        save_imgs(images_dec, ari_dir, 'ari_normal_inj_dec_%s' % config_name, nrow=4)

        smiley_enc_common = (torch.softmax(smiley_enc.mean(dim=0, keepdim=True) * 100, dim=1) > 0.9).to(torch.float32)
        smiley_common_dec, _ = dec(smiley_enc_common)
        save_imgs(smiley_common_dec, ari_dir, 'ari_smiley_common_dec_%s' % config_name, nrow=1)
        inj_mask = (torch.rand((1, 1, normal_enc_common.size(2)), device=device) > 0.35).to(torch.float32)
        lat_dec = other_enc.clone()
        for i in range(len(other_emos_ids)):
            lat_dec[i:i + 1, ...] = inj_mask * smiley_enc_common + (1 - inj_mask) * lat_dec[i:i + 1, ...]

        images_dec, _ = dec(lat_dec)
        save_imgs(images_dec, ari_dir, 'ari_smiley_inj_dec_%s' % config_name, nrow=4)
