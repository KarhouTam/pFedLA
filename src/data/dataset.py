import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision.transforms import ToTensor


class MNISTDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None) -> None:
        self.transform = transform
        self.target_transform = target_transform
        _data = data
        _targets = targets
        if not isinstance(_data, torch.Tensor):
            if not isinstance(_data, np.ndarray):
                _data = ToTensor()(_data)
            else:
                _data = torch.tensor(_data)
        self.data = _data.float().unsqueeze(1)

        if not isinstance(_targets, torch.Tensor):
            _targets = torch.tensor(_targets)
        self.targets = _targets.long()

    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)


class CIFARDataset(Dataset):
    def __init__(self, data, targets, transform=None, target_transform=None) -> None:
        self.transform = transform
        self.target_transform = target_transform
        _data = data
        _targets = targets
        if not isinstance(_data, torch.Tensor):
            if not isinstance(_data, np.ndarray):
                _data = ToTensor()(_data)
            else:
                _data = torch.tensor(_data)
        self.data = torch.permute(_data, [0, -1, 1, 2]).float()
        if not isinstance(_targets, torch.Tensor):
            _targets = torch.tensor(_targets)
        self.targets = _targets.long()


    def __getitem__(self, index):
        data, targets = self.data[index], self.targets[index]

        if self.transform is not None:
            data = self.transform(self.data[index])

        if self.target_transform is not None:
            targets = self.target_transform(self.targets[index])

        return data, targets

    def __len__(self):
        return len(self.targets)

