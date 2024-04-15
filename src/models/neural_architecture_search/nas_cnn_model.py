import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from ...data.fashion_mnist import get_fashion_mnist_data, FashionMNISTDataset
from ...features.loss_function_relaxation import get_default_device
from ...train_run import to_device
from torch.utils.data import DataLoader
from ...train_run import DeviceDataLoader
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import math

params = {
    "batch_size": 100,
    "lr": 0.01,
    "weight_decay": 0,
    "optimizer": "Adam",
    "hidden_units": 200,
    "conv_layers": 1,
    "dense_layers": 1,
    "kernel_conv": 2,
    "kernel_pooling": 2,
    "stride_conv": 2,
    "stride_pooling": 2,
    "padding_conv": 2,
    "padding_pooling": 2,
    "dropout": 0.5,
    "features": 150,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)


class CNNModel(nn.Module):
    def __init__(self, input_shape: int, output_shape: int):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=params["hidden_units"],
                kernel_size=params["kernel_conv"],
                padding=params["padding_conv"],
                stride=params["stride_conv"],
            ),
            nn.BatchNorm2d(params["hidden_units"]),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=params["kernel_pooling"],
                stride=params["stride_pooling"],
                padding=params["padding_pooling"],
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=params["hidden_units"],
                out_channels=params["hidden_units"] * 2,
                kernel_size=params["kernel_conv"],
                padding=params["padding_conv"],
                stride=params["stride_conv"],
            ),
            nn.BatchNorm2d(params["hidden_units"] * 2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=params["kernel_pooling"],
                stride=params["stride_pooling"],
                padding=params["padding_pooling"],
            ),
        )
        self.conv = (
            nn.Sequential(self.conv1, self.conv2)
            if params["conv_layers"] == 2
            else self.conv1
        )
        self.out_size = (
            math.floor(
                (28 - params["kernel_conv"] + 2 * params["padding_conv"])
                / params["stride_conv"]
            )
            + 1
        )
        self.out_size = (
            math.floor(
                (
                    self.out_size
                    - params["kernel_pooling"]
                    + 2 * params["padding_pooling"]
                )
                / params["stride_pooling"]
            )
            + 1
        )
        if params["conv_layers"] == 2:
            self.out_size = (
                math.floor(
                    (self.out_size - params["kernel_conv"] + 2 * params["padding_conv"])
                    / params["stride_conv"]
                )
                + 1
            )
            self.out_size = (
                math.floor(
                    (
                        self.out_size
                        - params["kernel_pooling"]
                        + 2 * params["padding_pooling"]
                    )
                    / params["stride_pooling"]
                )
                + 1
            )
        self.classifier = (
            nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=params["hidden_units"]
                    * params["conv_layers"]
                    * self.out_size
                    * self.out_size,
                    out_features=4 * params["features"],
                ),
                nn.Dropout(params["dropout"]),
                nn.Linear(
                    in_features=4 * params["features"], out_features=params["features"]
                ),
                nn.Linear(in_features=params["features"], out_features=output_shape),
            )
            if params["dense_layers"] == 2
            else nn.Sequential(
                nn.Flatten(),
                nn.Linear(
                    in_features=params["hidden_units"]
                    * params["conv_layers"]
                    * self.out_size
                    * self.out_size,
                    out_features=params["features"],
                ),
                nn.Dropout(params["dropout"]),
                nn.Linear(in_features=params["features"], out_features=output_shape),
            )
        )

    def forward(self, x, x1=None):
        x = self.conv(x)
        x = self.classifier(x)
        return x


def train_epoch(model, train_loader, optimizer, epoch, bin=True):
    loss_fn = nn.BCEWithLogitsLoss() if bin else nn.CrossEntropyLoss()
    total = 0
    sum_loss = 0
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for _, x1, x2, y, difficulties in tepoch:
            if 1 in list(y.shape) and not bin:
                y = y.squeeze(dim=1)
            tepoch.set_description(f"Epoch {epoch+1}")
            batch = y.shape[0]
            optimizer.zero_grad()
            output = model(x1, x2)

            loss = loss_fn(output, y)

            loss.backward()
            optimizer.step()
            total += batch
            sum_loss += batch * (loss.item())
            tepoch.set_postfix(loss=sum_loss / total)


def test_epoch(model, test_loader, bin=True):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y, difficulties in test_loader:
            current_batch_size = y.shape[0]
            if 1 in list(y.shape) and not bin:
                y = y.squeeze(dim=1)
            output = model(x1, x2)
            if not bin:
                y_pred_softmax = F.log_softmax(output, dim=1)
                _, y_pred_tags = torch.max(y_pred_softmax, dim=1)
                correct += (y_pred_tags == y.squeeze()).float().sum()
            else:
                correct += (F.sigmoid(output).round() == y).float().sum()
            total += current_batch_size

    accuracy = 100.0 * correct / total

    print("\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(correct, total, accuracy))

    return accuracy


def evaluate_model(model, train_dl, test_dl):
    device = get_default_device()
    to_device(model, device)
    optim_params = filter(lambda p: p.requires_grad, model.parameters())
    if params["optimizer"] == "Adam":
        optimizer = torch.optim.Adam(
            optim_params, lr=params["lr"], weight_decay=params["weight_decay"]
        )
    else:
        optimizer = torch.optim.SGD(
            optim_params, lr=params["lr"], weight_decay=params["weight_decay"]
        )

    # epochs: 5
    for epoch in range(5):
        train_epoch(model, train_dl, optimizer, epoch, bin=False)
        accuracy = test_epoch(model, test_dl, bin=False)
        print(accuracy.item())
        nni.report_intermediate_result(accuracy.item())

    nni.report_final_result(accuracy.item())


if __name__ == "__main__":

    device = get_default_device()
    train_data, test_data = get_fashion_mnist_data()
    X_train, y_train = map(list, zip(*[[x[0].numpy(), x[1]] for x in train_data]))
    X_test, y_test = map(list, zip(*[[x[0].numpy(), x[1]] for x in test_data]))

    train_ds = FashionMNISTDataset(X_train, y_train)
    test_ds = FashionMNISTDataset(X_test, y_test)

    batch_size = params["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    model = CNNModel(1, 10)

    evaluate_model(model, train_dl, test_dl)
