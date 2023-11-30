from typing import Any
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
from dotenv import load_dotenv

load_dotenv()
PATH = os.environ.get('PATH_TO_FILE')

class AirlinePassengersDataset(Dataset):

    def __init__(self, transform=None) -> None:
        """
        Args:
            transform (callable, optional): Optional transform to be applied on a sample
        """
        xy = pd.read_csv(PATH, delimiter=',', header=0)
        self.x = xy.iloc[:, 1:-1].to_numpy()
        self.y = xy.iloc[:, [-1]].to_numpy()
        self.n_samples = xy.shape[0]

        self.transform = transform

    def __getitem__(self, index: int):
        sample = self.x[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)
        
        return sample
    
    def __len__(self):
        return self.n_samples
    
class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)
    
# dataset = AirlinePassengersDataset(transform=None)
# first_data = dataset[0]

# features, labels = first_data
# print(features)
# print(len(dataset))
# print(type(features), type(labels))
