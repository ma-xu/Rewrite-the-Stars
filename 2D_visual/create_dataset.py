"""
Create 2D points dataset for visualization.

Created by: Xu Ma (Email: ma.xu1@northeastern.edu)
Modified Date: Mar/29/2024
"""

from sklearn.datasets import make_moons
import torch
from torch.utils.data import Dataset


class MoonDataset(Dataset):
    def __init__(self, n_samples=100, noise=0.2, random_state=0):
        dataset = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
        self.data = dataset[0]
        self.labels = dataset[1]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        y = torch.LongTensor([self.labels[idx]])
        return x, y


class ValDataset(Dataset):
    def __init__(self):
        range_w = torch.arange(-2, 3, step=0.02)
        range_h = torch.arange(-2, 2.5, step=0.02)
        data = torch.stack(torch.meshgrid(range_w, range_h, indexing='ij'), dim=-1).float()
        data = data.reshape(-1, 2)
        self.data = data

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        x = torch.Tensor(self.data[idx])
        return x
