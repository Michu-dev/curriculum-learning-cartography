import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from torch.utils.data import DataLoader
from tqdm import tqdm
from .loss_function_relaxation import relax_loss
import mlflow
from sklearn.metrics import roc_auc_score


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


def training_gnn_loop(
    epochs: int,
    model: GeneralisedNeuralNetworkModel,
    optimizer: torch.optim.Adam,
    train_dl: DeviceDataLoader,
    valid_dl: DeviceDataLoader,
    relaxed: bool = False,
    bin: bool = True,
) -> GeneralisedNeuralNetworkModel:
    for i in range(epochs):
        loss = train_gnn_model(model, optimizer, train_dl, i, relaxed=relaxed, bin=bin)
        val_loss, acc, all_preds, all_labels = validate_gnn_loss(
            model, valid_dl, i, relaxed=relaxed, bin=bin
        )

        mlflow.log_metric("train_loss", loss, step=i)
        mlflow.log_metric("validation_loss", val_loss, step=i)
        mlflow.log_metric("Accuracy", acc, step=i)

        if bin:
            auc = roc_auc_score(all_labels, all_preds)
            mlflow.log_metric("roc_auc", auc, step=i)

    return model


def train_gnn_model(
    model: GeneralisedNeuralNetworkModel,
    optim: torch.optim.Adam,
    train_dl: DataLoader,
    epoch: int,
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
    with tqdm(train_dl, unit="batch") as tepoch:
        for _, x1, x2, y, difficulties in tepoch:
            if 1 in list(y.shape) and not bin:
                y = y.squeeze(dim=1)
            tepoch.set_description(f"Epoch {epoch+1}")
            batch = y.shape[0]
            output = model(x1, x2)

            loss = loss_fn(output, y)

            if relaxed:
                loss = relax_loss(loss, difficulties, epoch + 1)

            optim.zero_grad()
            loss.backward()
            optim.step()
            total += batch
            sum_loss += batch * (loss.item())

            tepoch.set_postfix(loss=sum_loss / total)

    return sum_loss / total


def validate_gnn_loss(
    model: GeneralisedNeuralNetworkModel,
    valid_dl: DataLoader,
    epoch: int,
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

    with tqdm(valid_dl, unit="batch") as tepoch:
        for _, x1, x2, y, difficulties in tepoch:
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
                loss = relax_loss(loss, difficulties, epoch + 1)

            sum_loss += current_batch_size * (loss.item())
            total += current_batch_size
            correct += (preds == y).float().sum()

            tepoch.set_postfix(
                loss=sum_loss / total, accuracy=(correct / total).cpu().item()
            )

    all_labels = np.squeeze(np.array(all_labels).astype(int)).tolist()
    all_preds = np.squeeze(np.array(all_preds).astype(int))
    return sum_loss / total, correct / total, all_preds, all_labels