import numpy as np
from torch.utils.data import Dataset
import pandas as pd
import os
from PIL import Image
import torch


class CaptchaDataset(Dataset):
    def __init__(self, img_dir, annotations_file=None,  transform=None, **kwargs):
        super(CaptchaDataset, self).__init__(**kwargs)
        if annotations_file is not None:
            self.annotations = pd.read_csv(annotations_file, names=['file_name', 'label'])
        else:
            self.annotations = None
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return 20000

    def __getitem__(self, idx):
        if self.annotations is not None:
            img_path = os.path.join(self.img_dir, self.annotations.iloc[idx, 0])
        else:
            img_path = os.path.join(self.img_dir, f'{idx:05d}.jpg')
        image = Image.open(img_path)
        if self.transform:
            image = self.transform(image)
        # label = self.annotations.iloc[idx, 1]
        label = torch.randint(9, [4])
        return image, label

    @staticmethod
    def encode_label(target):
        encoded = torch.zeros(4, 10)
        for i, v in enumerate(f'{target:04d}'):
            v = int(v)
            encoded[i][v] = 1
        return encoded.flatten()

    @staticmethod
    def decode_label(target):
        decoded = target.reshape(4, 10)
        for i in decoded:
            v = i.argmax()
            decoded[v][i] = 1
        return decoded.flatten()