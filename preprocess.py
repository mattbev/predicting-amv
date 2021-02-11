import os
import torch
import numpy as np
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils


class Shift(object):
    """Shift [-1,1] to [0,2]"""
    def __call__(self, sample):
        data, label = sample["data"], sample["label"]
        return {
            "data" : data,
            "label" : label + 1
        }

class ToTensor(object):
    """Convert ndarrays in sample to Tensors."""

    def __call__(self, sample):
        data, label = sample['data'], sample['label']
        return {
            'data' : torch.from_numpy(data),
            'label' : torch.from_numpy(label).long().squeeze()
        }


class AMVDataset(Dataset):
    def __init__(self, istrain, percent_train, root_dir, ens, lead, tstep=86, transform=None):
        """
        Args:
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.transform = transform

        datapath = os.path.join(root_dir, f"CESM_data_sst_sss_psl_deseason_normalized_resized_detrend{lead}.npy")
        data = np.load(datapath)[:, 0:ens, ...]
        data = (data[:,:,:tstep-lead,:,:]).reshape(3,ens*(tstep-lead),224,224).transpose(1,0,2,3)
        if istrain:
            self.data = data[0:int(np.floor(percent_train*(tstep-lead)*ens)),:,:,:]
        else:
            self.data = data[int(np.floor(percent_train*(tstep-lead)*ens)):,:,:,:]

        labelpath = os.path.join(root_dir, f"CESM_label_amv_index_detrend{lead}.npy")
        label = np.load(labelpath)[0:ens, :]
        label = label[:ens,lead:].reshape(ens*(tstep-lead),1)
        if istrain:
            self.label = label[0:int(np.floor(percent_train*(tstep-lead)*ens)),:]
        else:
            self.label = label[int(np.floor(percent_train*(tstep-lead)*ens)):,:]

    def __len__(self):
        return self.label.size

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        sample = {
            'data': self.data[idx],
            'label': self.label[idx]
        }

        if self.transform:
            sample = self.transform(sample)

        return sample
