from ..data.airline_passenger_satisfaction_train import preprocess_airline_data, AirlinePassengersDataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from torch.utils.tensorboard import SummaryWriter
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from .auxiliary_functions import get_default_device, get_optimizer, to_device, DeviceDataLoader
# from data.airline_passenger_satisfaction_train import read_data, data_preprocessing, AirlinePassengersDataset, ToTensor
import pandas as pd

    
def train_gnn_model(model: GeneralisedNeuralNetworkModel, optim: torch.optim.Adam, train_dl: DataLoader) -> float:
    model.train()
    total = 0
    sum_loss = 0
    for x1, x2, y in train_dl:
        batch = y.shape[0]
        output = model(x1, x2)
        loss = F.binary_cross_entropy(output, y)
        optim.zero_grad()
        loss.backward()
        optim.step()
        total += batch
        sum_loss += batch * (loss.item())


    return sum_loss / total

def validate_gnn_loss(model: GeneralisedNeuralNetworkModel, valid_dl: DataLoader) -> tuple[float, float]:
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    for x1, x2, y in valid_dl:
        current_batch_size = y.shape[0]
        out = model(x1, x2)
        loss = F.binary_cross_entropy(out, y)
        sum_loss += current_batch_size * (loss.item())
        total += current_batch_size
        correct += (out.round() == y).float().sum()
    print("valid loss %.3f and accuracy %.3f " % (sum_loss / total, correct / total))
    return sum_loss / total, correct / total


def train_nn_airline(train_df: pd.DataFrame, embedded_cols: dict, epochs: int=8, batch_size: int= 1000, lr: float=0.01, wd: float=0.0) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    print(embedded_cols)
    embedding_sizes = [(n_categories+1, min(50, (n_categories + 1) // 2)) for _, n_categories in embedded_cols.items()]
    X, y = train_df.iloc[:, :-1], train_df.iloc[:, [-1]]
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=0)

    train_ds = AirlinePassengersDataset(X_train, y_train, embedded_col_names)
    valid_ds = AirlinePassengersDataset(X_val, y_val, embedded_col_names)

    device = get_default_device()
    model = GeneralisedNeuralNetworkModel(embedding_sizes, 7)
    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    for i in range(epochs):
        loss = train_gnn_model(model, optim, train_dl)
        print("training loss: %.3f" % loss)
        validate_gnn_loss(model, valid_dl)

    return model

def test_nn_airline(model: GeneralisedNeuralNetworkModel, test_df: pd.DataFrame, embedded_cols: dict, batch_size: int= 1000):
    embedded_col_names = embedded_cols.keys()
    X, y = test_df.iloc[:, :-1], test_df.iloc[:, [-1]]
    test_ds = AirlinePassengersDataset(X, y, embedded_col_names)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)

    preds = []
    with torch.no_grad():
        for x1, x2, y in test_dl:
            out = model(x1, x2)
            preds.append(out)
    
    final_probs = [item for sublist in preds for item in sublist]
    return final_probs


def main():
    train_df, test_df, embedded_cols = preprocess_airline_data()
    model = train_nn_airline(train_df, embedded_cols)
    final_probs = test_nn_airline(model, test_df, embedded_cols)
    # print(final_probs)


if __name__ == '__main__':
    main()

