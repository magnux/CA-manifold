import numpy as np
from multiprocessing.pool import ThreadPool
from tqdm import tqdm
from file_utils import download_url

url_base = 'https://github.com/googlefonts/noto-emoji/raw/master/png/128/emoji_u%s.png'


def download_emoji(entry):
    e_class = entry[2].lower().replace(' ', '_').replace('&', 'and')
    e_code = entry[4].lower().replace(' ', '_')
    if e_code is not None:
        try:
            download_url(url_base % e_code, 'data/emoji/%s' % e_class)
        except Exception as e:
            pass
    return e_code


emoji_array = np.fromregex('src/utils/emoji_df.csv', r'(.+),(.+),(.+),(.+),(.+)', np.object)
emoji_list = emoji_array[1:, ...].tolist()

print('Downloading Emojis...')

results = ThreadPool(16).imap_unordered(download_emoji, emoji_list)
for result in tqdm(results, total=len(emoji_list)):
    result
