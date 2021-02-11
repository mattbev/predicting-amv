import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        return {'data': torch.from_numpy(data),
                'label': torch.from_numpy(label).long()}
    

class AMVDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.root_dir = root_dir
        self.transform = transform

    def __len__(self):
        return len(next(os.walk(self.root_dir))[1])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        datapath = os.path.join(
            self.root_dir,
            str(idx),
            "x.npy"
        )
        labelpath = os.path.join(
            self.root_dir,
            str(idx),
            "y.npy"
        )
        data = np.load(datapath).squeeze()
        label = np.load(labelpath)
        label = label.squeeze()
        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
