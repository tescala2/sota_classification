import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None, target_transform=None):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        self.transform = transform
        self.target_transform = target_transform

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        if self.transform:
            image = self.transform(image)
        if self.target_transform:
            label = self.target_transform(label)
        return image, label


def get_loaders(path, bs, shuffle, transform=None, target_transform=None):
    loaders = []
    for mode in ['train', 'val', 'test']:
        annotations_path = f'{path}/{mode}/annotations.csv'
        img_path = f'{path}/{mode}/images'

        data = ImageDataset(annotations_path, img_path, transform=transform, target_transform=target_transform)
        loaders.append(DataLoader(data, batch_size=bs, shuffle=shuffle))

    return loaders
