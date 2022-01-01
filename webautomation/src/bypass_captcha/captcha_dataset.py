from torch.utils.data import Dataset
import pandas as pd
import os
from torchvision.io import read_image


class CaptchaDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file, names=['file_name', 'label'])
        self.img_dir = img_dir
        self.transform = transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.decode_label(label)
        return image, label

    @staticmethod
    def decode_label(target):
        decoded = [int(j) for j in str(target)]
        return decoded
