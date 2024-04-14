import torch
import torch.nn as nn
import torch.nn.functional as F
import nni
from nni.nas.nn.pytorch import LayerChoice, ModelSpace, MutableDropout, MutableLinear
import nni.nas.strategy as strategy
from nni.nas.evaluator import FunctionalEvaluator
from nni.nas.experiment import NasExperiment
from ...data.stellar_ds import preprocess_stellar_ds, StellarDataset
from ...features.loss_function_relaxation import get_default_device
from ...train_run import to_device, get_optimizer
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from ...train_run import DeviceDataLoader
from tqdm import tqdm
import time


class SeparableLinear(nn.Module):
    def __init__(self, in_num, out_num, drops):
        super().__init__()
        feature = nni.choice("feature", [50, 100, 150, 250])
        self.lin1 = MutableLinear(in_num, feature)
        self.lin2 = MutableLinear(feature, out_num)
        self.bn = nn.BatchNorm1d(self.lin2.in_features)
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
        self.drops = MutableDropout(nni.choice("dropout", [0.25, 0.5, 0.75]))
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = LayerChoice(
            [nn.Linear(200, n_class), SeparableLinear(200, n_class, self.drops)],
            label="linear2",
        )
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.emb_drop = nn.Dropout(0.6)

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


def train_epoch(model, train_loader, optimizer, epoch):
    loss_fn = torch.nn.CrossEntropyLoss()
    total = 0
    sum_loss = 0
    model.train()
    with tqdm(train_loader, unit="batch") as tepoch:
        for _, x1, x2, y, difficulties in tepoch:
            if 1 in list(y.shape):
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


def test_epoch(model, test_loader):
    model.eval()
    total, correct = 0, 0
    with torch.no_grad():
        for _, x1, x2, y, difficulties in test_loader:
            current_batch_size = y.shape[0]
            if 1 in list(y.shape):
                y = y.squeeze(dim=1)
            output = model(x1, x2)
            y_pred_softmax = F.log_softmax(output, dim=1)
            _, y_pred_tags = torch.max(y_pred_softmax, dim=1)

            total += current_batch_size
            correct += (y_pred_tags == y.squeeze()).float().sum()

    accuracy = 100.0 * correct / total

    print("\nTest set: Accuracy: {}/{} ({:.0f}%)\n".format(correct, total, accuracy))

    return accuracy


def evaluate_model(model):
    device = get_default_device()
    to_device(model, device)
    optimizer = get_optimizer(model, lr=0.01, wd=0.0)
    X_train, X_test, y_train, y_test, embedded_cols = preprocess_stellar_ds()
    embedded_col_names = embedded_cols.keys()
    batch_size = 100
    train_ds = StellarDataset(X_train, y_train, embedded_col_names)
    test_ds = StellarDataset(X_test, y_test, embedded_col_names)
    train_dl = DataLoader(train_ds, batch_size=batch_size, shuffle=True)
    test_dl = DataLoader(test_ds, batch_size=batch_size, shuffle=True)
    train_dl = DeviceDataLoader(train_dl, device)
    test_dl = DeviceDataLoader(test_dl, device)

    for epoch in range(10):
        train_epoch(model, train_dl, optimizer, epoch)
        accuracy = test_epoch(model, test_dl)
        print(accuracy.item())
        nni.report_intermediate_result(accuracy.item())

    nni.report_final_result(accuracy.item())


if __name__ == "__main__":
    X_train, X_test, y_train, y_test, embedded_cols = preprocess_stellar_ds()
    embedded_col_names = embedded_cols.keys()

    batch_size = 100
    # X_train, X_val, y_train, y_val = train_test_split(
    #     X_rem, y_rem, test_size=0.20, random_state=0
    # )
    embedding_sizes = [
        (n_categories + 1, min(50, (n_categories + 1) // 2))
        for _, n_categories in embedded_cols.items()
    ]

    n_cont = len(X_train.columns) - len(embedded_cols)

    model_space = FeedForwardNeuralNetworkModelSpace(embedding_sizes, n_cont, n_class=3)
    search_strategy = strategy.Random(dedup=False)
    evaluator = FunctionalEvaluator(evaluate_model)
    exp = NasExperiment(model_space, evaluator, search_strategy)
    exp.config.experiment_name = "StellarNAS"
    exp.config.trial_concurrency = 3
    exp.config.max_trial_number = 24
    exp.config.training_service.use_active_gpu = False
    # exp.config.trial_gpu_number = 1
    # exp.config.training_service.use_active_gpu = True
    exp.run(port=8081)
    for model_dict in exp.export_top_models(formatter="dict", top_k=3):
        print(model_dict)

    time.sleep(1000000)
