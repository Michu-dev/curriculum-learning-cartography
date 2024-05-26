import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear
from ...data.stellar_ds import preprocess_stellar_ds, StellarDataset
from ...data.airline_passenger_satisfaction_train import (
    preprocess_airline_data,
    AirlinePassengersDataset,
)
from ...data.credit_card_fraud import preprocess_credit_card_ds, CreditCardDataset
from ...data.spotify_tracks_genre import (
    preprocess_spotify_tracks_ds,
    SpotifyTracksDataset,
)
from ...features.loss_function_relaxation import get_default_device
from ...train_run import to_device
from torch.utils.data import DataLoader
from ...train_run import DeviceDataLoader
from tqdm import tqdm
from imblearn.over_sampling import SMOTE
import time
import sys


params = {
    "batch_size": 100,
    "lr": 0.01,
    "weight_decay": 0,
    "optimizer": "Adam",
    "hidden_layers": 1,
    "dropout": 0.5,
    "emb_dropout": 0.25,
    "features": 150,
}
optimized_params = nni.get_next_parameter()
params.update(optimized_params)


class SeparableLinear(nn.Module):
    def __init__(self, in_num, out_num, drops):
        super().__init__()
        feature = params["features"]
        self.lin1 = MutableLinear(in_num, feature)
        self.lin2 = MutableLinear(feature, out_num)
        self.bn = nn.BatchNorm1d(params["features"])
        self.drops = drops

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn(x)
        return self.lin2(x)


class FeedForwardNeuralNetworkModelSpace(ModelSpace):
    def __init__(self, embedding_sizes, n_cont, n_class=1):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for categories, size in embedding_sizes]
        )
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.drops = nn.Dropout(params["dropout"])
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = (
            nn.Linear(200, n_class)
            if params["hidden_layers"] <= 1
            else SeparableLinear(200, n_class, self.drops)
        )
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.emb_drop = nn.Dropout(params["emb_dropout"])

    def forward(self, x_cat, x_cont):
        x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
        if len(x) != 0:
            x = torch.cat(x, 1)
            x = self.emb_drop(x)
            x2 = self.bn1(x_cont)
            x = torch.cat([x, x2], 1)
        else:
            x = self.bn1(x_cont.float())
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn2(x)
        x = self.lin2(x)
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


def evaluate_model(model, train_dl, test_dl, bin=True):
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
        train_epoch(model, train_dl, optimizer, epoch, bin=bin)
        accuracy = test_epoch(model, test_dl, bin=bin)
        print(accuracy.item())
        nni.report_intermediate_result(accuracy.item())

    nni.report_final_result(accuracy.item())


if __name__ == "__main__":

    dataset = sys.argv[1]
    device = get_default_device()
    binary = True
    match dataset:
        case "airline_passenger_satisfaction":
            train_df, test_df, embedded_cols = preprocess_airline_data()
            embedded_col_names = embedded_cols.keys()
            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]
            X_train, y_train = train_df.iloc[:, :-1], train_df.iloc[:, [-1]]
            X_test, y_test = test_df.iloc[:, :-1], test_df.iloc[:, [-1]]
            train_ds = AirlinePassengersDataset(X_train, y_train, embedded_col_names)
            test_ds = AirlinePassengersDataset(X_test, y_test, embedded_col_names)
            model = FeedForwardNeuralNetworkModelSpace(embedding_sizes, 7)
        case "credit_card":
            X_train, X_test, y_train, y_test, prop = preprocess_credit_card_ds()
            smote = SMOTE(random_state=42)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            train_ds = CreditCardDataset(X_train, y_train)
            test_ds = CreditCardDataset(X_test, y_test)
            model = FeedForwardNeuralNetworkModelSpace([], len(X_train[0]))
        case "spotify_tracks":
            X_train, X_test, y_train, y_test, embedded_cols = (
                preprocess_spotify_tracks_ds()
            )
            embedded_col_names = embedded_cols.keys()
            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]
            train_ds = SpotifyTracksDataset(X_train, y_train, embedded_col_names)
            test_ds = SpotifyTracksDataset(X_test, y_test, embedded_col_names)
            n_cont = len(X_train.columns) - len(embedded_cols)
            model = FeedForwardNeuralNetworkModelSpace(
                embedding_sizes, n_cont, n_class=114
            )
            binary = False
        case "stellar" | _:
            X_train, X_test, y_train, y_test, embedded_cols = preprocess_stellar_ds()
            embedded_col_names = embedded_cols.keys()
            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]
            train_ds = StellarDataset(X_train, y_train, embedded_col_names)
            test_ds = StellarDataset(X_test, y_test, embedded_col_names)
            n_cont = len(X_train.columns) - len(embedded_cols)
            model = FeedForwardNeuralNetworkModelSpace(
                embedding_sizes, n_cont, n_class=3
            )
            binary = False

    batch_size = params["batch_size"]
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=False)
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    evaluate_model(model, train_dl, test_dl, bin=binary)
