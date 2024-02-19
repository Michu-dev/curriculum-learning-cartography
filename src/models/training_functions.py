from ..data.airline_passenger_satisfaction_train import AirlinePassengersDataset
from ..data.credit_card_fraud import CreditCardDataset
from ..data.spotify_tracks_genre import SpotifyTracksDataset
from ..data.stellar_ds import StellarDataset
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, WeightedRandomSampler
import numpy as np
import pandas as pd
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from .loss_function_relaxation import get_default_device
from .auxiliary_functions import (
    get_optimizer,
    to_device,
    train_gnn_model,
    validate_gnn_loss,
    data_cartography,
    plot_cartography_map,
    DeviceDataLoader,
)
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
from sklearn.model_selection import train_test_split
import mlflow
import tensorflow as tf
from pathlib import Path


def training_gnn_loop(
    epochs: int,
    model: GeneralisedNeuralNetworkModel,
    optimizer: torch.optim.Adam,
    train_dl: DeviceDataLoader,
    valid_dl: DeviceDataLoader,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    bin: bool = True,
) -> GeneralisedNeuralNetworkModel:
    for i in range(epochs):
        loss = train_gnn_model(
            model, optimizer, train_dl, i, skorch_model, relaxed=relaxed, bin=bin
        )
        val_loss, acc, all_preds, all_labels = validate_gnn_loss(
            model, valid_dl, i, skorch_model, relaxed=relaxed, bin=bin
        )

        mlflow.log_metric("train_loss", loss, step=i)
        mlflow.log_metric("validation_loss", val_loss, step=i)
        mlflow.log_metric("Accuracy", acc, step=i)

        if bin:
            auc = roc_auc_score(all_labels, all_preds)
            mlflow.log_metric("roc_auc", auc, step=i)

    return model


def train_nn_airline(
    train_df: pd.DataFrame,
    embedded_cols: dict,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    print(embedded_cols)
    embedding_sizes = [
        (n_categories + 1, min(50, (n_categories + 1) // 2))
        for _, n_categories in embedded_cols.items()
    ]
    X, y = train_df.iloc[:, :-1], train_df.iloc[:, [-1]]
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=0
    )

    train_ds = AirlinePassengersDataset(X_train, y_train, embedded_col_names)
    valid_ds = AirlinePassengersDataset(X_val, y_val, embedded_col_names)

    device = get_default_device()
    model = GeneralisedNeuralNetworkModel(embedding_sizes, 7)
    model_for_cartography = GeneralisedNeuralNetworkModel(embedding_sizes, 7)

    to_device(model, device)
    to_device(model_for_cartography, device)
    optim_for_cartography = get_optimizer(model_for_cartography, lr=lr, wd=wd)
    optim = get_optimizer(model, lr=lr, wd=wd)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    loss_fn = torch.nn.BCEWithLogitsLoss()

    path_to_save_training_dynamics = Path("./") / "airline_training_dynamics"

    cartography_stats_df = data_cartography(
        model_for_cartography,
        train_dl,
        loss_fn,
        optim_for_cartography,
        epochs,
        path_to_save_training_dynamics,
        binary=True,
    )

    print(cartography_stats_df)

    if plot_map:
        plot_cartography_map(
            cartography_stats_df,
            path_to_save_training_dynamics,
            "Airline_passengers_satisfaction",
            True,
        )

    # keras_data_cartography_params(
    #     model, X_train, y_train, X_val, y_val, embedded_col_names, binary=True
    # )

    model = training_gnn_loop(
        epochs, model, optim, train_dl, valid_dl, skorch_model, relaxed=relaxed
    )

    return model


def test_nn_airline(
    model: GeneralisedNeuralNetworkModel,
    test_df: pd.DataFrame,
    embedded_cols: dict,
    batch_size: int = 1000,
) -> list:
    embedded_col_names = embedded_cols.keys()
    X, y = test_df.iloc[:, :-1], test_df.iloc[:, [-1]]
    test_ds = AirlinePassengersDataset(X, y, embedded_col_names)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)

    preds = []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y in test_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)
            preds.append(out)

            total += current_batch_size
            correct += (F.sigmoid(out).round() == y).float().sum()

    mlflow.log_metric("test_acc", (correct / total))

    final_probs = [item for sublist in preds for item in sublist]
    return final_probs


def train_nn_spotify_tracks(
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    print(embedded_cols)
    embedding_sizes = [
        (n_categories + 1, min(50, (n_categories + 1) // 2))
        for _, n_categories in embedded_cols.items()
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=0
    )

    train_ds = SpotifyTracksDataset(X_train, y_train, embedded_col_names)
    valid_ds = SpotifyTracksDataset(X_val, y_val, embedded_col_names)

    device = get_default_device()
    n_cont = len(X.columns) - len(embedded_cols)
    # n_class based on previous EDA
    model = GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=114)
    to_device(model, device)

    optim = get_optimizer(model, lr=lr, wd=wd)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = training_gnn_loop(
        epochs,
        model,
        optim,
        train_dl,
        valid_dl,
        skorch_model,
        relaxed=relaxed,
        bin=False,
    )

    return model


def test_nn_spotify_tracks(
    model: GeneralisedNeuralNetworkModel,
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    batch_size: int = 1000,
):
    embedded_col_names = embedded_cols.keys()
    test_ds = SpotifyTracksDataset(X, y, embedded_col_names)

    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)

    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y in test_dl:
            current_batch_size = y.shape[0]
            y = y.squeeze(dim=1)
            out = model(x1, x2)
            y_pred_softmax = F.log_softmax(out, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            all_preds.extend(y_pred_tags.cpu().detach().numpy().astype(int).tolist())
            all_labels.extend(y.cpu().detach().numpy().astype(int).tolist())

            total += current_batch_size

            correct += (y_pred_tags == y.squeeze()).float().sum()

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int)).tolist()

    # metrics for imbalanced data model
    # precision = precision_score(all_labels, all_preds)
    # recall = recall_score(all_labels, all_preds)
    # f1score = f1_score(all_labels, all_preds)

    # mlflow.log_metric("test_acc", (correct / total))
    # mlflow.log_metric("precision", precision)
    # mlflow.log_metric("recall", recall)
    # mlflow.log_metric("f1_score", f1score)

    print("test accuracy %.3f " % (correct / total))
    return float(correct) / total


def train_nn_credit_card(
    X: pd.DataFrame,
    y: pd.DataFrame,
    propotions: list,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, stratify=y, random_state=0
    )

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

    train_sampler = WeightedRandomSampler(
        sample_train_weights, num_samples=len(sample_train_weights), replacement=True
    )

    train_dl = DataLoader(train_ds, batch_size=batch_size, sampler=train_sampler)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = training_gnn_loop(
        epochs,
        model,
        optim,
        train_dl,
        valid_dl,
        skorch_model,
        relaxed=relaxed,
    )

    return model


def test_nn_credit_card(
    model: GeneralisedNeuralNetworkModel,
    X: pd.DataFrame,
    y: pd.DataFrame,
    batch_size: int = 1000,
) -> float:
    test_ds = CreditCardDataset(X, y)
    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)
    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y in test_dl:
            current_batch_size = y.shape[0]
            out = model(x1, x2)

            all_preds.extend(
                F.sigmoid(out).round().cpu().detach().numpy().astype(int).tolist()
            )
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


def train_nn_stellar(
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    print(embedded_cols)
    embedding_sizes = [
        (n_categories + 1, min(50, (n_categories + 1) // 2))
        for _, n_categories in embedded_cols.items()
    ]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=0
    )

    train_ds = StellarDataset(X_train, y_train, embedded_col_names)
    valid_ds = StellarDataset(X_val, y_val, embedded_col_names)

    device = get_default_device()
    n_cont = len(X.columns) - len(embedded_cols)
    # n_class based on previous EDA
    model = GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=3)
    model_for_cartography = GeneralisedNeuralNetworkModel(
        embedding_sizes, n_cont, n_class=3
    )
    to_device(model, device)
    to_device(model_for_cartography, device)

    optim = get_optimizer(model, lr=lr, wd=wd)
    optim_for_cartography = get_optimizer(model_for_cartography, lr=lr, wd=wd)

    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    loss_fn = torch.nn.CrossEntropyLoss()

    path_to_save_training_dynamics = Path("./") / "stellar_training_dynamics"

    cartography_stats_df = data_cartography(
        model_for_cartography,
        train_dl,
        loss_fn,
        optim_for_cartography,
        epochs,
        path_to_save_training_dynamics,
        binary=False,
    )

    if plot_map:
        plot_cartography_map(
            cartography_stats_df, path_to_save_training_dynamics, "Stellar", True
        )

    model = training_gnn_loop(
        epochs,
        model,
        optim,
        train_dl,
        valid_dl,
        skorch_model,
        relaxed=relaxed,
        bin=False,
    )

    return model


def test_nn_stellar(
    model: GeneralisedNeuralNetworkModel,
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    batch_size: int = 1000,
):
    embedded_col_names = embedded_cols.keys()
    test_ds = StellarDataset(X, y, embedded_col_names)

    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)

    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y in test_dl:
            current_batch_size = y.shape[0]
            if 1 in list(y.shape):
                y = y.squeeze(dim=1)
            out = model(x1, x2)
            y_pred_softmax = F.log_softmax(out, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            all_preds.extend(y_pred_tags.cpu().detach().numpy().astype(int).tolist())
            all_labels.extend(y.cpu().detach().numpy().astype(int).tolist())

            total += current_batch_size

            correct += (y_pred_tags == y.squeeze()).float().sum()

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int)).tolist()

    # metrics for imbalanced data model
    # precision = precision_score(all_labels, all_preds)
    # recall = recall_score(all_labels, all_preds)
    # f1score = f1_score(all_labels, all_preds)

    mlflow.log_metric("test_acc", (correct / total))
    # mlflow.log_metric("precision", precision)
    # mlflow.log_metric("recall", recall)
    # mlflow.log_metric("f1_score", f1score)

    print("test accuracy %.3f " % (correct / total))
    return float(correct) / total
