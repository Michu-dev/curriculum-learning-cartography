import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from torch.utils.data import DataLoader
from skorch import NeuralNetClassifier
from tqdm import tqdm
from cleanlab.rank import get_self_confidence_for_each_label
from .loss_function_relaxation import relax_loss
from functorch.compile import draw_graph


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
        for x1, x2, y in tepoch:
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
        for x1, x2, y in tepoch:
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
