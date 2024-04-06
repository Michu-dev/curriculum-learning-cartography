from .data.airline_passenger_satisfaction_train import AirlinePassengersDataset
from .data.credit_card_fraud import CreditCardDataset
from .data.spotify_tracks_genre import SpotifyTracksDataset
from .data.stellar_ds import StellarDataset
from .data.fashion_mnist import FashionMNISTDataset
from .features.cartography_functions import data_cartography, plot_cartography_map
from .models.generalised_neural_network_model import GeneralisedNeuralNetworkModel
from .models.cnn_classifier import FashionMNISTModel
from .features.loss_function_relaxation import get_default_device
from .train_run import (
    get_optimizer,
    to_device,
    training_gnn_loop,
    DeviceDataLoader,
)
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader, Subset, Dataset
import numpy as np
import pandas as pd
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from skorch.helper import SliceDict
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split, cross_val_predict
from sklearn.preprocessing import MinMaxScaler
import umap
from sklearn.cluster import KMeans
import mlflow
from pathlib import Path
from cleanlab.rank import get_self_confidence_for_each_label
from imblearn.over_sampling import SMOTE
import logging
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import adjusted_mutual_info_score, adjusted_rand_score

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def get_self_confidence_rank_and_difficulties(
    X: dict,
    y: np.ndarray,
    model: GeneralisedNeuralNetworkModel | FashionMNISTModel,
    dataset: Dataset,
    loss_fn,
    epochs: int,
    batch_size: int,
    lr: float,
) -> Subset:
    skorch_model = NeuralNetClassifier(
        model,
        criterion=loss_fn,
        max_epochs=epochs,
        batch_size=batch_size,
        lr=lr,
    )
    X_skorch = SliceDict(**X)
    pred_probs = cross_val_predict(
        skorch_model,
        X_skorch,
        y,
        cv=5,
        method="predict_proba",
    )
    y_qualities = get_self_confidence_for_each_label(
        y.astype(np.int64), pred_probs
    ).squeeze()
    difficulties = np.ones_like(y_qualities) - y_qualities
    examples_order = np.argsort(difficulties)
    dataset.difficulties = difficulties

    # if not relaxed:
    dataset = Subset(dataset, indices=examples_order)
    return dataset


def get_cartography_rank_and_difficulties(
    model: GeneralisedNeuralNetworkModel | FashionMNISTModel,
    path: Path,
    dataset: Dataset,
    loss_fn,
    plot_flag: bool,
    plot_title: str,
    binary: bool,
    epochs: int,
    device: torch.device,
    batch_size: int,
    lr: float,
    wd: float,
) -> Subset:
    to_device(model, device)
    optim_for_cartography = get_optimizer(model, lr=lr, wd=wd)
    cartography_train_dl = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    cartography_train_dl = DeviceDataLoader(cartography_train_dl, device)
    cartography_stats_df = data_cartography(
        model,
        cartography_train_dl,
        loss_fn,
        optim_for_cartography,
        epochs,
        path,
        binary=binary,
    )
    # clustering -> assign examples to classes: easy-to-learn/ambiguous/hard-to-learn
    # dimensionality reduction -> UMAP 2D -> 1D with higher classification maintenance | normalized weights of 2D features
    main_label, other_metric, cluster_num = "variability", "confidence", "labels"
    x_train = cartography_stats_df[[main_label, other_metric]]
    x_train[main_label] = 2 * x_train[main_label]
    kmeans_labels = KMeans(n_clusters=3, random_state=42, n_init="auto").fit_predict(
        x_train
    )
    cartography_stats_df["labels"] = kmeans_labels
    if plot_flag:
        plot_cartography_map(
            cartography_stats_df,
            path,
            plot_title,
            True,
            model.__class__.__name__,
        )
        # Plot clustering results
        sns.set(style="whitegrid", font_scale=1.6, font="Georgia", context="paper")
        fig, ax0 = plt.subplots(1, 1, figsize=(10, 8))
        pal = sns.diverging_palette(260, 15, n=3, sep=10, center="dark")
        plot = sns.scatterplot(
            x=main_label,
            y=other_metric,
            ax=ax0,
            data=cartography_stats_df,
            hue=cluster_num,
            palette=pal,
            style=cluster_num,
            s=30,
        )
        plot.set_xlabel("variability")
        plot.set_ylabel("confidence")
        plot.set_title(f"{plot_title}-Cartography K-Means result", fontsize=17)
        path = path / "clustering_results"
        if not path.exists():
            path.mkdir(parents=True, exist_ok=True)
        fig.tight_layout()
        fig.savefig(f"{path}/{plot_title}.pdf")

    y_qualities = np.ones(len(dataset), dtype=np.float32)

    # embedding = umap.UMAP(n_components=1, random_state=42).fit_transform(
    #     x_train, kmeans_labels
    # )

    # Interpolation mode - to consider
    # embedding = (x_train[other_metric] + x_train[main_label]).values.reshape(-1, 1)
    # embedding = MinMaxScaler().fit_transform(embedding)

    # reduced_kmeans_labels = KMeans(
    #     n_clusters=3, random_state=0, n_init="auto"
    # ).fit_predict(embedding)
    # embedding = embedding.reshape(-1)

    # Loop matching distributed training dynamics values with dataset
    for i, x in enumerate(dataset):
        curr_row_idx = cartography_stats_df.index[cartography_stats_df["guid"] == x[0]]
        y_qualities[i] = cartography_stats_df.loc[curr_row_idx, other_metric]

    print(dataset[:10])
    print(cartography_stats_df[cartography_stats_df["guid"].between(0, 9)])
    print(y_qualities[:10])

    # logger.info(
    #     f"Adjusted_rand_score: {adjusted_rand_score(kmeans_labels, reduced_kmeans_labels)}"
    # )
    # logger.info(
    #     f"Adjusted mutual information: {adjusted_mutual_info_score(kmeans_labels, reduced_kmeans_labels)}"
    # )

    examples_order = np.argsort(y_qualities)[::-1]
    dataset.difficulties = y_qualities

    # To consider
    # if not relaxed:
    dataset = Subset(dataset, indices=examples_order)
    return dataset


def train_nn_airline(
    train_df: pd.DataFrame,
    embedded_cols: dict,
    rank_mode: str,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    logger.info(f"Embedded column names (categorical): {embedded_cols}")
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
    if rank_mode == "confidence":
        X1 = X_train.loc[:, embedded_col_names].copy().values.astype(np.int64)
        X2 = X_train.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        y = y_train.copy().values.astype(np.float32)
        X = {
            "x_cat": X1,
            "x_cont": X2,
        }
        train_ds = get_self_confidence_rank_and_difficulties(
            X,
            y,
            GeneralisedNeuralNetworkModel(embedding_sizes, 7),
            train_ds,
            torch.nn.BCEWithLogitsLoss,
            epochs,
            batch_size,
            lr,
        )
    elif rank_mode == "cartography":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        path_to_save_training_dynamics = Path("./") / "airline_training_dynamics"

        train_ds = get_cartography_rank_and_difficulties(
            GeneralisedNeuralNetworkModel(embedding_sizes, 7),
            path_to_save_training_dynamics,
            train_ds,
            loss_fn,
            plot_map,
            "Airline_passengers_satisfaction",
            True,
            epochs,
            device,
            batch_size,
            lr,
            wd,
        )

    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    if rank_mode:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    valid_dl = DataLoader(valid_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    valid_dl = DeviceDataLoader(valid_dl, device)

    model = training_gnn_loop(epochs, model, optim, train_dl, valid_dl, relaxed=relaxed)

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
    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y, _ in test_dl:
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

    precision = precision_score(all_labels, all_preds)
    recall = recall_score(all_labels, all_preds)
    f1score = f1_score(all_labels, all_preds)

    mlflow.log_metric("test_acc", (correct / total))
    mlflow.log_metric("precision", precision)
    mlflow.log_metric("recall", recall)
    mlflow.log_metric("f1_score", f1score)

    logger.info("test accuracy %.3f " % (correct / total))
    return float(correct) / total


def train_nn_spotify_tracks(
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    rank_mode: str,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    logger.info(f"Embedded column names (categorical): {embedded_cols}")
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

    if rank_mode == "confidence":
        X1 = X_train.loc[:, embedded_col_names].copy().values.astype(np.int64)
        X2 = X_train.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        y = y_train.copy().values.astype(np.int64).squeeze()
        X = {
            "x_cat": X1,
            "x_cont": X2,
        }
        n_cont = len(X_train.columns) - len(embedded_cols)
        train_ds = get_self_confidence_rank_and_difficulties(
            X,
            y,
            GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=114),
            train_ds,
            torch.nn.CrossEntropyLoss,
            epochs,
            batch_size,
            lr,
        )
    elif rank_mode == "cartography":
        loss_fn = torch.nn.CrossEntropyLoss()
        path_to_save_training_dynamics = (
            Path("./") / "spotify_tracks_genre_training_dynamics"
        )

        train_ds = get_cartography_rank_and_difficulties(
            GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=114),
            path_to_save_training_dynamics,
            train_ds,
            loss_fn,
            plot_map,
            "Spotify_tracks_genre",
            False,
            epochs,
            device,
            batch_size,
            lr,
            wd,
        )

    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    if rank_mode:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
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
        for _, x1, x2, y, _ in test_dl:
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

    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")
    f1score = f1_score(all_labels, all_preds, average="micro")

    mlflow.log_metric("test_acc", (correct / total))
    mlflow.log_metric("micro_precision", precision)
    mlflow.log_metric("micro_recall", recall)
    mlflow.log_metric("micro_f1_score", f1score)

    logger.info("test accuracy %.3f " % (correct / total))
    return float(correct) / total


def train_nn_credit_card(
    X: pd.DataFrame,
    y: pd.DataFrame,
    propotions: list,
    rank_mode: str,
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

    # Apply SMOTE oversampling to the training data
    smote = SMOTE(random_state=42)
    X_train, y_train = smote.fit_resample(X_train, y_train)

    train_ds = CreditCardDataset(X_train, y_train)
    valid_ds = CreditCardDataset(X_val, y_val)
    device = get_default_device()
    model = GeneralisedNeuralNetworkModel([], len(X_train[0]))

    if rank_mode == "confidence":

        X = X_train.copy().astype(np.float32)
        y = y_train.copy().values.astype(np.float32)

        X = {
            "x_cat": np.zeros(X.shape[0], dtype=np.float32),
            "x_cont": X,
        }
        train_ds = get_self_confidence_rank_and_difficulties(
            X,
            y,
            GeneralisedNeuralNetworkModel([], len(X_train[0])),
            train_ds,
            torch.nn.BCEWithLogitsLoss,
            epochs,
            batch_size,
            lr,
        )
    elif rank_mode == "cartography":
        loss_fn = torch.nn.BCEWithLogitsLoss()
        path_to_save_training_dynamics = Path("./") / "card_fraud_training_dynamics"

        train_ds = get_cartography_rank_and_difficulties(
            GeneralisedNeuralNetworkModel([], len(X_train[0])),
            path_to_save_training_dynamics,
            train_ds,
            loss_fn,
            plot_map,
            "Credit_card_fraud_detection",
            True,
            epochs,
            device,
            batch_size,
            lr,
            wd,
        )

    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    if rank_mode:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
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
        for _, x1, x2, y, _ in test_dl:
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

    logger.info("test accuracy %.3f " % (correct / total))
    return float(correct) / total


def train_nn_stellar(
    X: pd.DataFrame,
    y: pd.DataFrame,
    embedded_cols: dict,
    rank_mode: str,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    embedded_col_names = embedded_cols.keys()
    logger.info(f"Embedded column names (categorical): {embedded_cols}")
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

    if rank_mode == "confidence":
        X1 = X_train.loc[:, embedded_col_names].copy().values.astype(np.int64)
        X2 = X_train.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        y = y_train.copy().values.astype(np.int64).squeeze()
        X = {
            "x_cat": X1,
            "x_cont": X2,
        }
        n_cont = len(X_train.columns) - len(embedded_cols)
        train_ds = get_self_confidence_rank_and_difficulties(
            X,
            y,
            GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=3),
            train_ds,
            torch.nn.CrossEntropyLoss,
            epochs,
            batch_size,
            lr,
        )
    elif rank_mode == "cartography":
        loss_fn = torch.nn.CrossEntropyLoss()
        path_to_save_training_dynamics = Path("./") / "stellar_training_dynamics"

        train_ds = get_cartography_rank_and_difficulties(
            GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=3),
            path_to_save_training_dynamics,
            train_ds,
            loss_fn,
            plot_map,
            "Stellar",
            False,
            epochs,
            device,
            batch_size,
            lr,
            wd,
        )

    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    if rank_mode:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
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
        for _, x1, x2, y, _ in test_dl:
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

    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")
    f1score = f1_score(all_labels, all_preds, average="micro")

    mlflow.log_metric("test_acc", (correct / total))
    mlflow.log_metric("micro_precision", precision)
    mlflow.log_metric("micro_recall", recall)
    mlflow.log_metric("micro_f1_score", f1score)

    logger.info("test accuracy %.3f " % (correct / total))
    return float(correct) / total


def train_cnn_fashion_mnist(
    X: torch.Tensor,
    y: torch.Tensor,
    rank_mode: str,
    relaxed: bool = False,
    epochs: int = 8,
    batch_size: int = 1000,
    lr: float = 0.01,
    plot_map: bool = False,
    wd: float = 0.0,
) -> GeneralisedNeuralNetworkModel:
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.10, random_state=0
    )

    train_ds = FashionMNISTDataset(X_train, y_train)
    valid_ds = FashionMNISTDataset(X_val, y_val)

    device = get_default_device()
    # n_class - 10 for Fashin MNIST
    model = FashionMNISTModel(1, 32, 10)

    if rank_mode == "confidence":
        X = np.asarray(X_train, dtype=np.float32)
        y = np.asarray(y_train, dtype=np.int64)
        X = {
            "x": X,
            "x1": np.ones(X.shape),
        }
        train_ds = get_self_confidence_rank_and_difficulties(
            X,
            y,
            FashionMNISTModel(1, 32, 10),
            train_ds,
            torch.nn.CrossEntropyLoss,
            epochs,
            batch_size,
            lr,
        )
    elif rank_mode == "cartography":
        loss_fn = torch.nn.CrossEntropyLoss()
        path_to_save_training_dynamics = Path("./") / "fashion_mnist_training_dynamics"

        train_ds = get_cartography_rank_and_difficulties(
            FashionMNISTModel(1, 32, 10),
            path_to_save_training_dynamics,
            train_ds,
            loss_fn,
            plot_map,
            "Fashion_Mnist",
            False,
            epochs,
            device,
            batch_size,
            lr,
            wd,
        )

    to_device(model, device)
    optim = get_optimizer(model, lr=lr, wd=wd)

    if rank_mode:
        train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=False)
    else:
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
        relaxed=relaxed,
        bin=False,
    )

    return model


def test_cnn_fashion_mnist(
    model: FashionMNISTDataset,
    X: torch.Tensor,
    y: torch.Tensor,
    batch_size: int = 1000,
):
    test_ds = FashionMNISTDataset(X, y)

    test_dl = DataLoader(test_ds, batch_size=batch_size)
    device = get_default_device()
    test_dl = DeviceDataLoader(test_dl, device)

    all_preds, all_labels = [], []
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y, _ in test_dl:
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

    precision = precision_score(all_labels, all_preds, average="micro")
    recall = recall_score(all_labels, all_preds, average="micro")
    f1score = f1_score(all_labels, all_preds, average="micro")

    mlflow.log_metric("test_acc", (correct / total))
    mlflow.log_metric("micro_precision", precision)
    mlflow.log_metric("micro_recall", recall)
    mlflow.log_metric("micro_f1_score", f1score)

    logger.info("test accuracy %.3f " % (correct / total))
    return float(correct) / total
