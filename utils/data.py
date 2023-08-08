import os
import pandas as pd

from torch.utils.data import Dataset, DataLoader
from torchvision.io import read_image


class ImageDataset(Dataset):
    def __init__(self, annotations_file, img_dir, transform=None):
        self.img_labels = pd.read_csv(annotations_file)
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
        return image, label


def get_dataloader(path, mode, bs, transform, shuffle=True, num_workers=24):
    dataset = ImageDataset(
        annotations_file=f'{path}/data/{mode}/chips/labels.csv',
        img_dir=f'{path}/data/{mode}/chips/images/',
        transform=transform
    )
    dataloader = DataLoader(dataset,
                            batch_size=bs,
                            shuffle=shuffle,
                            num_workers=num_workers,
                            pin_memory=True)

    return dataloader
