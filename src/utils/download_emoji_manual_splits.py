import numpy as np
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from file_utils import download_url
import yaml

url_base = 'https://github.com/googlefonts/noto-emoji/raw/main/png/128/%s'


def download_emoji(sm):
    split, emoji = sm
    try:
        download_url(url_base % emoji, 'data/%s/%s' % (split, emoji))
    except Exception as e:
        pass
    return emoji


with open('src/utils/emoji_manual_splits.yaml', 'r') as f:
    splits = yaml.load(f, Loader=yaml.FullLoader)

emoji_list = [(split, emoji) for split in splits.keys() for emoji in splits[split]]

print('Downloading Emojis...')

results = ThreadPool(16).imap_unordered(download_emoji, emoji_list)
for result in tqdm(results, total=len(emoji_list)):
    result
