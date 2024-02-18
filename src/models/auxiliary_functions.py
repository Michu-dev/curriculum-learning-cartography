import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from tqdm import tqdm
from cleanlab.rank import get_self_confidence_for_each_label
from .loss_function_relaxation import relax_loss
from functorch.compile import draw_graph
import tensorflow as tf
import nobuco
from nobuco import ChannelOrder
import tavolo as tvl
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import logging
import json

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def to_device(data, device):
    """Move tensor(s) to chosen device"""
    if isinstance(data, (list, tuple)):
        return [to_device(x, device) for x in data]
    return data.to(device, non_blocking=True)


def get_optimizer(model, lr=0.001, wd=0):
    parameters = filter(lambda p: p.requires_grad, model.parameters())
    optim = torch.optim.SGD(parameters, lr=lr, weight_decay=wd)
    return optim


class DeviceDataLoader:
    """Wrap a dataloader to move data to a device"""

    def __init__(self, dl, device):
        self.dl = dl
        self.device = device

    def __iter__(self):
        """Yield a batch of data after moving it to device"""
        for b in self.dl:
            yield to_device(b, self.device)

    def __len__(self):
        """Number of batches"""
        return len(self.dl)


def cartography_map_plot(
    confidence: np.ndarray, variability: np.ndarray, correctness: np.ndarray
):
    _, ax = plt.subplots(figsize=(9, 7))

    sns.scatterplot(x=variability, y=confidence, hue=correctness, ax=ax)
    sns.kdeplot(
        x=variability,
        y=confidence,
        levels=8,
        color=sns.color_palette("Paired")[7],
        linewidths=1,
        ax=ax,
    )

    # dataset details to implement
    ax.set(
        title="Data map for airline_passenger_satisfaction train set\nbased on a feed-forward neural net classifier",
        xlabel="Variability",
        ylabel="Confidence",
    )

    # Annotations
    box_style = {"boxstyle": "round", "facecolor": "white", "ec": "black"}
    ax.text(
        0.14,
        0.84,
        "easy-to-learn",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=box_style,
    )
    ax.text(
        0.75,
        0.5,
        "ambiguous",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=box_style,
    )
    ax.text(
        0.14,
        0.14,
        "hard-to-learn",
        transform=ax.transAxes,
        verticalalignment="top",
        bbox=box_style,
    )

    ax.legend(title="Correctness")

    plt.show()


def log_training_dynamics(
    output_dir: Path,
    epoch: int,
    train_ids: List[int],
    train_logits: List[List[float]],
    train_golds: List[int],
):
    """
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """
    td_df = pd.DataFrame(
        {
            "guid": train_ids,
            f"logits_epoch_{epoch}": train_logits,
            "gold": train_golds,
        }
    )
    logging_dir = output_dir / "training_dynamics"
    if not logging_dir.exists():
        logging_dir.mkdir(parents=True, exist_ok=True)
    epoch_file_name = logging_dir / f"dynamics_epoch_{epoch}.jsonl"
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Training Dynamics logged to {epoch_file_name}")


def read_training_dynamics(model_dir: Path) -> dict:
    train_dynamics = {}
    td_dir = model_dir / "training_dynamics"

    epochs = len([f for f in td_dir.iterdir() if f.is_file()])
    logger.info(f"Reading {epochs} files from {td_dir} ...")

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_file = td_dir / f"dynamics_epoch_{epoch}.jsonl"
        with epoch_file.open() as md_file:
            content = md_file.readlines()
            for line in content:
                record = json.loads(line.strip())
                guid = record["guid"]
                if guid not in train_dynamics:
                    train_dynamics[guid] = {"gold": record["gold"], "logits": []}
                train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch}"])

    logger.info(f"Read training dynamics for {len(train_dynamics)} train instances.")
    return train_dynamics


def data_cartography(
    model: GeneralisedNeuralNetworkModel,
    train_dl: DeviceDataLoader,
    loss_fn,
    optim,
    epochs: int,
    path_to_save_training_dynamics: Path,
    binary: bool = False,
) -> pd.DataFrame:

    if not path_to_save_training_dynamics.exists():
        path_to_save_training_dynamics.mkdir(parents=True, exist_ok=True)
        model.train()

        # think about adding eval set
        for epoch in range(epochs):
            total, sum_loss = 0, 0
            train_ids, train_golds, train_logits = None, None, None
            with tqdm(train_dl, unit="batch") as tepoch:
                for i, (idx, x1, x2, y) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1}")

                    if 1 in list(y.shape) and not binary:
                        y = y.squeeze(dim=1)

                    batch = y.shape[0]
                    output = model(x1, x2)
                    loss = loss_fn(output, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total += batch
                    sum_loss += batch * (loss.item())

                    if train_logits is None:
                        train_ids = idx.detach().cpu().numpy()
                        train_logits = output.detach().cpu().numpy()
                        train_golds = y.detach().cpu().numpy()
                    else:
                        train_ids = np.append(train_ids, idx.detach().cpu().numpy())
                        train_logits = np.append(
                            train_logits, output.detach().cpu().numpy(), axis=0
                        )
                        train_golds = np.append(train_golds, y.detach().cpu().numpy())

                    tepoch.set_postfix(loss=sum_loss / total)

            log_training_dynamics(
                path_to_save_training_dynamics,
                epoch + 1,
                list(train_ids),
                list(train_logits),
                list(train_golds),
            )

    training_dynamics = read_training_dynamics(path_to_save_training_dynamics)
    confidence_ = {}
    variability_ = {}
    correctness_ = {}

    for guid in tqdm(training_dynamics):
        correctness_trend = []
        true_probs_trend = []

        record = training_dynamics[guid]
        for epoch_logits in record["logits"]:
            probs = (
                torch.nn.functional.sigmoid(torch.Tensor(epoch_logits))
                if binary
                else torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
            )
            probs = torch.Tensor([1.0 - probs[0], probs[0]]) if binary else probs
            true_class_prob = float(probs[int(record["gold"])])
            true_probs_trend.append(true_class_prob)

            prediction = np.argmax(probs)
            is_correct = (prediction == record["gold"]).item()
            correctness_trend.append(is_correct)

        correctness_[guid] = sum(correctness_trend)
        confidence_[guid] = np.mean(true_probs_trend)
        variability_[guid] = np.std(true_probs_trend)

    column_names = [
        "guid",
        "confidence",
        "variability",
        "correctness",
    ]
    df = pd.DataFrame(
        [
            [
                guid,
                confidence_[guid],
                variability_[guid],
                correctness_[guid],
            ]
            for guid in correctness_
        ],
        columns=column_names,
    )

    return df


# def keras_data_cartography_params(
#     model: GeneralisedNeuralNetworkModel,
#     X_train: pd.DataFrame,
#     y_train: pd.DataFrame,
#     X_val: pd.DataFrame,
#     y_val: pd.DataFrame,
#     embedded_col_names: list,
#     binary: bool = False,
# ):
#     X1_keras = X_train.loc[:, embedded_col_names].copy().values.astype(np.int64)
#     X2_keras = X_train.drop(columns=embedded_col_names).copy().values.astype(np.float32)
#     y_keras = y_train.copy().values.astype(np.int32)

#     X1_keras_val = X_val.loc[:, embedded_col_names].copy().values.astype(np.int64)
#     X2_keras_val = (
#         X_val.drop(columns=embedded_col_names).copy().values.astype(np.float32)
#     )
#     y_keras_val = y_val.copy().values.astype(np.int32)

#     keras_ds_x1 = tf.data.Dataset.from_tensor_slices(X1_keras)
#     keras_ds_x2 = tf.data.Dataset.from_tensor_slices(X2_keras)
#     keras_ds_y = tf.data.Dataset.from_tensor_slices(y_keras)
#     keras_ds_x = tf.data.Dataset.zip((keras_ds_x1, keras_ds_x2))
#     keras_ds = tf.data.Dataset.zip((keras_ds_x, keras_ds_y))

#     keras_ds_x1 = tf.data.Dataset.from_tensor_slices(X1_keras_val)
#     keras_ds_x2 = tf.data.Dataset.from_tensor_slices(X2_keras_val)
#     keras_ds_y = tf.data.Dataset.from_tensor_slices(y_keras_val)
#     keras_ds_x = tf.data.Dataset.zip((keras_ds_x1, keras_ds_x2))
#     keras_ds_val = tf.data.Dataset.zip((keras_ds_x, keras_ds_y))
#     # print(tuple(zip(*keras_ds.take(1))))
#     # X1_train, X2_train, y_train = keras_ds.map(lambda x, y, z: (x, y, z))
#     # X1_val, X2_val, y_val = keras_ds_val.map(lambda x, y, z: (x, y, z))

#     keras_model = nobuco.pytorch_to_keras(
#         model,
#         args=[torch.tensor(X1_keras), torch.tensor(X2_keras)],
#         inputs_channel_order=ChannelOrder.TENSORFLOW,
#         outputs_channel_order=ChannelOrder.TENSORFLOW,
#     )
#     print("asdasdasdasd")
#     keras_ds_shuffled = keras_ds.shuffle(len(y_keras)).batch(100)

#     dataset_map = tvl.learning.DataMapCallback(
#         keras_ds.batch(100),
#         outputs_to_probabilities=lambda x: (
#             tf.nn.sigmoid(x[0]) if binary else tf.nn.softmax(x[0])
#         ),  # Model outputs a tuple, where the logits are at the first index
#         sparse_labels=not binary,
#     )
#     optimizer = tf.keras.optimizers.Adam(learning_rate=1e-4)
#     loss = tf.keras.losses.BinaryCrossentropy(from_logits=True)

#     # callbacks = [dataset_map, tf.keras.callbacks.EarlyStopping(patience=5)]
#     callbacks = [dataset_map]

#     keras_model.compile(optimizer=optimizer, loss=loss, metrics=["accuracy"])
#     keras_model.fit(
#         keras_ds_shuffled,
#         # [X1_keras, X2_keras],
#         # y_keras,
#         epochs=10,
#         validation_data=keras_ds_val.batch(32),
#         steps_per_epoch=150,
#         callbacks=callbacks,
#     )

#     print(dataset_map.confidence)
#     print(dataset_map.variability)
#     print(dataset_map.correctness)
#     print("----------------")
#     print(set(dataset_map.correctness))
#     print("----------------")
#     print(set(dataset_map.confidence))
#     print("----------------")

#     cartography_map_plot(
#         dataset_map.confidence, dataset_map.variability, dataset_map.correctness
#     )


def train_gnn_model(
    model: GeneralisedNeuralNetworkModel,
    optim: torch.optim.Adam,
    train_dl: DataLoader,
    epoch: int,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    bin: bool = True,
) -> float:
    model.train()
    total = 0
    sum_loss = 0
    print("Training:")
    reduction = "none" if relaxed else "mean"
    loss_fn = (
        nn.BCEWithLogitsLoss(reduction=reduction)
        if bin
        else nn.CrossEntropyLoss(reduction=reduction)
    )
    # start_idx = 0
    # y_train_qualities = y_train_qualities.squeeze(2)
    with tqdm(train_dl, unit="batch") as tepoch:
        for _, x1, x2, y in tepoch:
            if 1 in list(y.shape) and not bin:
                y = y.squeeze(dim=1)
            tepoch.set_description(f"Epoch {epoch+1}")
            batch = y.shape[0]
            output = model(x1, x2)
            # print(output.shape)
            # print("--------------------")
            # print(y.shape)
            loss = loss_fn(output, y)

            if relaxed:
                X1_train = x1.cpu().detach().numpy().astype(np.int64)
                X2_train = x2.cpu().detach().numpy().astype(np.float32)

                y_train_clean = y.cpu().detach().numpy().copy().astype(np.int32)

                X_train_clean = {
                    "x_cat": X1_train,
                    "x_cont": X2_train,
                }

                pred_probs = skorch_model.predict_proba(X_train_clean)

                y_train_qualities = get_self_confidence_for_each_label(
                    y_train_clean, pred_probs
                )
                # print(y_train_qualities)

                if bin:
                    difficulty = y_train_qualities.squeeze(2)
                else:
                    difficulty = np.expand_dims(y_train_qualities, axis=1)
                difficulty = np.ones_like(difficulty) - difficulty

                # loss = loss_fn(output, y, difficulty, epoch + 1)

                loss = relax_loss(loss, difficulty, epoch + 1)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())
            # start_idx += batch

            tepoch.set_postfix(loss=sum_loss / total)

    return sum_loss / total


def validate_gnn_loss(
    model: GeneralisedNeuralNetworkModel,
    valid_dl: DataLoader,
    epoch: int,
    skorch_model: NeuralNetClassifier,
    relaxed: bool = False,
    bin: bool = True,
) -> tuple[float, float, list, list]:
    model.eval()
    total = 0
    sum_loss = 0
    correct = 0
    all_preds, all_labels = [], []
    print("Validation:")
    reduction = "none" if relaxed else "mean"
    loss_fn = (
        nn.BCEWithLogitsLoss(reduction=reduction)
        if bin
        else nn.CrossEntropyLoss(reduction=reduction)
    )
    # start_idx = 0
    # y_val_qualities = y_val_qualities.squeeze(2)
    with tqdm(valid_dl, unit="batch") as tepoch:
        for _, x1, x2, y in tepoch:
            if 1 in list(y.shape) and not bin:
                y = y.squeeze(dim=1)
            tepoch.set_description(f"Epoch {epoch+1}")
            current_batch_size = y.shape[0]
            out = model(x1, x2)

            preds = F.sigmoid(out).round() if bin else F.log_softmax(out, dim=1)
            if not bin:
                _, preds = torch.max(preds, dim=1)

            all_preds.extend(preds.cpu().detach().numpy().astype(int).tolist())
            all_labels.extend(y.cpu().detach().numpy().astype(int).tolist())

            loss = loss_fn(out, y)

            if relaxed:
                X1_val = x1.cpu().detach().numpy().astype(np.int64)
                X2_val = x2.cpu().detach().numpy().astype(np.float32)
                y_val_clean = y.cpu().detach().numpy().astype(np.int32)

                X_val_clean = {
                    "x_cat": X1_val,
                    "x_cont": X2_val,
                }

                pred_probs = skorch_model.predict_proba(X_val_clean)

                y_val_qualities = get_self_confidence_for_each_label(
                    y_val_clean, pred_probs
                )

                # print(y_val_qualities[:10])

                if bin:
                    difficulty = y_val_qualities.squeeze(2)
                else:
                    difficulty = np.expand_dims(y_val_qualities, axis=1)

                difficulty = np.ones_like(difficulty) - difficulty
                # indexes = np.where(difficulty < 0.5)[0]
                # print(indexes)
                # print(difficulty[indexes, :])
                # print("---------------")
                # print(preds[indexes, :])
                # print("---------------")
                # print(y[indexes, :])
                # loss = loss_fn(out, y, difficulty, epoch + 1)
                loss = relax_loss(loss, difficulty, epoch + 1)

            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
            correct += (preds == y).float().sum()
            # start_idx += current_batch_size

            tepoch.set_postfix(
                loss=sum_loss / total, accuracy=(correct / total).cpu().item()
            )

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int))
    return sum_loss / total, correct / total, all_preds, all_labels
