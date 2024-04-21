import os
import torch
import numpy as np
from torch.utils.data import Dataset
from torchvision import datasets
from torchvision.transforms import ToTensor
from dotenv import load_dotenv
from typing import Tuple

load_dotenv()


def get_fashion_mnist_data() -> (
    Tuple[Tuple[torch.Tensor, torch.Tensor], Tuple[torch.Tensor, torch.Tensor]]
):
    train_data = datasets.FashionMNIST(
        root=os.environ.get("PATH_TO_FASHION_MNIST"),
        train=True,
        download=True,
        transform=ToTensor(),
        target_transform=None,
    )

    test_data = datasets.FashionMNIST(
        root=os.environ.get("PATH_TO_FASHION_MNIST"),
        train=False,
        download=True,
        transform=ToTensor(),
    )

    return train_data, test_data


class FashionMNISTDataset(Dataset):

    def __init__(self, X, y, transform=None) -> None:
        """
        Args:
            X: input data
            y: labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        X = X.copy()
        self.X = np.asarray(X, dtype=np.float32)
        self.y = np.asarray(y, dtype=np.int64)
        self.id = np.arange(len(self.y))
        self.difficulties = np.zeros(len(self.y), dtype=np.int64)

        self.transform = transform

    def __getitem__(self, index: int):
        sample = (
            self.id[index],
            self.X[index],
            0,
            self.y[index],
            self.difficulties[index],
        )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.y)
