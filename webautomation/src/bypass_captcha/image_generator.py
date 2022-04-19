from pathlib import Path
import time
import requests
from image import ImageCaptcha
import numpy as np
from os import path as osp
import timeout_decorator
from PIL import Image, UnidentifiedImageError
import io

generator = ImageCaptcha()

# %% Generate demo images


def demo():
    # Generate one demo image
    for i, k in enumerate([6109]):
        print(f'{i:05d}_{k:04d}.png')
        image = generator.write(f'{k:04d}', osp.join('captcha_images', 'demo', f'{i:05d}_{k:04d}.png'))

    # Generate demo images
    v = np.random.randint(0, 9999, [50000])
    with open('captcha_labels.txt', 'w+') as f:
        for i, k in enumerate(v):
            print(f'{i:05d}_{k:04d}.png')
            image = generator.write(f'{k:04d}', osp.join('captcha_images', 'train', f'{i:05d}_{k:04d}.png'))
            f.write(f'{i:05d}_{k:04d}.png, {k:04d}\n')

    v2 = np.random.randint(0, 9999, [10000])
    with open('captcha_test_labels.txt', 'w+') as f2:
        for i, k in enumerate(v2):
            print(f'{i:05d}_{k:04d}.png')
            image = generator.write(f'{k:04d}', osp.join('captcha_images', 'test', f'{i:05d}_{k:04d}.png'))
            f2.write(f'{i:05d}_{k:04d}.png, {k:04d}\n')


# demo()
# %% Get images from url
@timeout_decorator.timeout(3)
def r():
    return requests.get("https://hk.sz.gov.cn:8118/user/getVerify?0.6619971119181707").content

dir = 'images'

def get():
    latest = int(sorted(list(Path(dir).iterdir()))[-1].stem)
    while True:
        try:
            img_data = r()
            try:
                Image.open(io.BytesIO(img_data))
                with open(f'{dir}/{latest + 1:05d}.jpg', 'wb') as handler:
                    handler.write(img_data)
                latest += 1
            except:
                time.sleep(np.random.rand(50)[0])
                continue
        except timeout_decorator.timeout_decorator.TimeoutError:
            time.sleep(np.random.rand(50)[0])
        time.sleep(np.random.rand(2)[0])



get()
print('finished')