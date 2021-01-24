import json
import os

import numpy as np
import torch
from PIL import Image
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from torchvision import transforms


class IMDB(Dataset):
    def __init__(self, root_dir, mode):
        super(IMDB, self).__init__()
        self.root_dir = root_dir
        self.transform = transforms.Compose(
            [
            transforms.ColorJitter(brightness=0.4, contrast=0.2,
                                    saturation=0.4, hue=0.05),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                std=[0.229, 0.224, 0.225])]
        )

        catalog_path = os.path.join(root_dir, mode + '.json')
        try:
            os.path.exists(catalog_path)
        except FileExistsError:
            print('catalog does not exist')
        self.catalog = json.load(open(catalog_path))
        self.catalog_keys = list(self.catalog.keys())
        self.device = torch.device(
            'cuda' if torch.cuda.is_available() else 'cpu')

    def __len__(self):
        return len(self.catalog)

    def __getitem__(self, idx):
        sample = self.catalog[self.catalog_keys[idx]]
        image = Image.open(os.path.join(self.root_dir, sample['image']))
        width, height = 256, 192
        image = image.resize((width, height))
        label = Image.open(os.path.join(self.root_dir, sample['label']))
        reduction = 2
        label = label.resize((int(width/reduction),
                              int(height/reduction)), Image.NEAREST)
        if self.transform:
            image = self.transform(image)
        return image, np.array(label).astype(np.long)


def imdb_loader(args):
    train_loader = DataLoader(dataset=IMDB(args.dataset_dir, mode='train'),
                              batch_size=args.batch_size, shuffle=True,
                              num_workers=8, drop_last=True)

    eval_loader = DataLoader(dataset=IMDB(args.dataset_dir, mode='eval'),
                             batch_size=args.batch_size, shuffle=False,
                             num_workers=8, drop_last=False)
    return train_loader, eval_loader