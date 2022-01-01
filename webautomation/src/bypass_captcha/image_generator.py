from image import ImageCaptcha
import numpy as np
from os import path as osp

v = np.random.randint(0, 9999, [10000])
generator = ImageCaptcha()

with open(osp.join('src', 'captcha_labels.txt'), 'w+') as f:
    f.write('file_name, label\n')
    for i, k in enumerate(v):
        print(f'{i:04d}_{k:04d}.png')
        image = generator.write(f'{k:04d}', osp.join('src', 'captcha_images', f'{i:04d}_{k:04d}.png'))
        f.write(f'{i:04d}_{k:04d}.png, {k:04d}\n')
