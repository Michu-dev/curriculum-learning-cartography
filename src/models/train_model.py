from ..data.airline_passenger_satisfaction_train import preprocess_airline_data, AirlinePassengersDataset
from ..data.credit_card_fraud import preprocess_credit_card_ds, CreditCardDataset
import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import matplotlib.pyplot as plt
import os
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from torch.utils.tensorboard import SummaryWriter
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from .auxiliary_functions import get_default_device, get_optimizer, to_device, DeviceDataLoader
# from data.airline_passenger_satisfaction_train import read_data, data_preprocessing, AirlinePassengersDataset, ToTensor
import pandas as pd
import numpy as np
import plac
import mlflow
from tqdm import tqdm



def train_gnn_model(model: GeneralisedNeuralNetworkModel, optim: torch.optim.Adam, train_dl: DataLoader, epoch: int) -> float:
    model.train()
    total = 0
    sum_loss = 0
    print("Training:")
    with tqdm(train_dl, unit="batch") as tepoch:
        for x1, x2, y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            batch = y.shape[0]
            output = model(x1, x2)
            
            loss = F.binary_cross_entropy_with_logits(output, y)
            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())

            tepoch.set_postfix(loss=sum_loss/total)


    return sum_loss / total

def validate_gnn_loss(model: GeneralisedNeuralNetworkModel, valid_dl: DataLoader, epoch: int) -> tuple[float, float, list, list]:
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    all_preds, all_labels = [], []
    print("Validation:")
    with tqdm(valid_dl, unit="batch") as tepoch:
        for x1, x2, y in tepoch:
            tepoch.set_description(f"Epoch {epoch}")
            current_batch_size = y.shape[0]
            out = model(x1, x2)
            
            all_preds.extend(F.sigmoid(out).round().cpu().detach().numpy().astype(int).tolist())
            all_labels.extend(y.cpu().detach().numpy().astype(int).tolist())

            loss = F.binary_cross_entropy_with_logits(out, y)
            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
            correct += (F.sigmoid(out).round() == y).float().sum()

            tepoch.set_postfix(loss=sum_loss/total, accuracy=(correct/total).cpu().item())

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int))
    return sum_loss / total, correct / total, all_preds, all_labels


def training_gnn_loop(epochs: int, model: GeneralisedNeuralNetworkModel, optimizer: torch.optim.Adam,
                       train_dl: DeviceDataLoader, valid_dl: DeviceDataLoader) -> GeneralisedNeuralNetworkModel:
    for i in range(epochs):
        loss = train_gnn_model(model, optimizer, train_dl, i)
        val_loss, acc, all_preds, all_labels = validate_gnn_loss(model, valid_dl, i)

        auc = roc_auc_score(all_labels, all_preds)
        mlflow.log_metric("train_loss", loss, step=i)
        mlflow.log_metric("validation_loss", val_loss, step=i)
        mlflow.log_metric("Accuracy", acc, step=i)
        mlflow.log_metric("roc_auc", auc, step=i)

    return model

def train_nn_airline(train_df: pd.DataFrame, embedded_cols: dict,
                     epochs: int=8, batch_size: int= 1000, lr: float=0.01, wd: float=0.0) -> GeneralisedNeuralNetworkModel:
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

    model = training_gnn_loop(epochs, model, optim, train_dl, valid_dl)

    return model

def test_nn_airline(model: GeneralisedNeuralNetworkModel, test_df: pd.DataFrame,
                     embedded_cols: dict, batch_size: int=1000) -> list:
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



def train_nn_credit_card(X: pd.DataFrame, y: pd.DataFrame, propotions: list, epochs: int=8,
                          batch_size: int= 1000, lr: float=0.01, wd: float=0.0) -> GeneralisedNeuralNetworkModel:
    
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.10, random_state=0)

    train_ds = CreditCardDataset(X_train, y_train)
    valid_ds = CreditCardDataset(X_val, y_val)
    device = get_default_device()
    model = GeneralisedNeuralNetworkModel([], len(X_train[0]))
    to_device(model, device)
    
    optim = get_optimizer(model, lr=lr, wd=wd)
    sample_train_weights = [0] * len(train_ds)

    for idx, (x1, x2, y) in enumerate(train_ds):
        class_weight = propotions[y[0].astype(int)]
        sample_train_weights[idx] = class_weight
    
    train_sampler = WeightedRandomSampler(sample_train_weights, num_samples=
                                          len(sample_train_weights), replacement=True)
    
    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = training_gnn_loop(epochs, model, optim, train_dl, valid_dl)
    
    return model

def test_nn_credit_card(model: GeneralisedNeuralNetworkModel, X: pd.DataFrame, y: pd.DataFrame, 
                        batch_size: int=1000) -> float:
    
    test_ds = CreditCardDataset(X, y)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)
    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for x1, x2, y in test_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)

            all_preds.extend(F.sigmoid(out).round().cpu().detach().numpy().astype(int).tolist())
            all_labels.extend(y.cpu().detach().numpy().astype(int).tolist())

            total += current_batch_size
            correct += (F.sigmoid(out).round() == y).float().sum()
    

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int)).tolist()
    
    # metrics for imbalanced data model
    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1score = f1_score(all_labels, all_preds)

    mlflow.log_metric("test_acc", (correct / total))
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1score)

    print("test accuracy %.3f " % (correct / total))
    return float(correct) / total

@plac.opt('dataset', 'Dataset to use for NN model evaluation', str, 'd')
@plac.opt('batch_size', 'Batch size of the data', int, 'b')
@plac.opt('epochs', 'Epochs number of training', int, 'e')
@plac.opt('lr', 'Learning rate of optimizer', float, 'l')
def main(dataset: str='credit_card', batch_size: int=1000, epochs: int=8, lr: float=0.01):
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        if dataset == 'airline_passenger_satisfaction':
            train_df, test_df, embedded_cols = preprocess_airline_data()
            model = train_nn_airline(train_df, embedded_cols, epochs=epochs, batch_size=batch_size, lr=lr)
            final_probs = test_nn_airline(model, test_df, embedded_cols, batch_size=batch_size)
        elif dataset == 'credit_card':
            X_rem, X_test, y_rem, y_test, prop = preprocess_credit_card_ds()
            model = train_nn_credit_card(X_rem, y_rem, prop, epochs=epochs,
                                            batch_size=batch_size, lr=lr)
            test_acc = test_nn_credit_card(model, X_test, y_test,
                                        batch_size=batch_size)
            
        mlflow.log_param('dataset', dataset)
        mlflow.log_param('batch_size', batch_size)
        mlflow.log_param('epochs', epochs)
        mlflow.log_param('learning_rate', lr)

        mlflow.pytorch.log_model(model, 'model')


if __name__ == '__main__':
    # main()
    plac.call(main)

