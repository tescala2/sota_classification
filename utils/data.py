import os
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, mode='train'):
        self.img_labels = pd.read_csv(annotations_file)
        self.img_dir = img_dir
        assert mode in ['train', 'val', 'test'], "Invalid mode, should be in {'train', 'val', 'test'}"
        if mode == 'train':
            self.transform = torch.nn.Sequential(transforms.Resize((224, 224)),
                                                 transforms.RandomRotation(degrees=(0, 359)),
                                                 transforms.RandomHorizontalFlip(p=0.5),
                                                 transforms.RandomVerticalFlip(p=0.5))
        if mode == 'val' or mode == 'test':
            self.transform = transforms.Resize((224, 224))

    def __len__(self):
        return len(self.img_labels)

    def __getitem__(self, idx):
        img_path = os.path.join(self.img_dir, self.img_labels.iloc[idx, 0])
        image = read_image(img_path)
        label = self.img_labels.iloc[idx, 1]
        image = self.transform(image)
        return image, label


def get_dataloader(path, mode, bs, shuffle=True, num_workers=24):
    dataset = ImageDataset(
        annotations_file=f'/data/{path}/{mode}/labels.csv',
        img_dir=f'/data/{path}/{mode}/images/',
        mode=mode
    )
    dataloader = DataLoader(dataset,
                            batch_size=bs,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True)

    return dataloader
