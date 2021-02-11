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
    def __init__(self, root_dir, ens, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.ens = ens
        self.transform = transform
        
        datapath = os.path.join(root_dir, "x.npy")
        labelpath = os.path.join(root_dir, "y.npy")
        self.data = np.load(datapath)[:, 0:ens, ...]
        self.label = np.load(labelpath)[0:ens, :]

    def __len__(self):
        return len(next(os.walk(self.root_dir))[1])

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {'data': data, 'label': label}

        if self.transform:
            sample = self.transform(sample)

        return sample
