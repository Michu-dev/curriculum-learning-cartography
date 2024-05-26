import torch
import torch.nn as nn
import torch.nn.functional as F


# class GeneralisedNeuralNetworkModel(nn.Module):
#     def __init__(self, embedding_sizes, n_cont, n_class=1):
#         super().__init__()
#         self.embeddings = nn.ModuleList(
#             [nn.Embedding(categories, size) for categories, size in embedding_sizes]
#         )
#         n_emb = sum(e.embedding_dim for e in self.embeddings)
#         self.n_emb, self.n_cont = n_emb, n_cont
#         self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
#         self.lin2 = nn.Linear(200, 70)
#         self.lin3 = nn.Linear(70, n_class)
#         self.bn1 = nn.BatchNorm1d(self.n_cont)
#         self.bn2 = nn.BatchNorm1d(200)
#         self.bn3 = nn.BatchNorm1d(70)
#         self.emb_drop = nn.Dropout(0.6)
#         self.drops = nn.Dropout(0.3)

#     def forward(self, x_cat, x_cont):
#         x = [e(x_cat[:, i]) for i, e in enumerate(self.embeddings)]
#         if len(x) != 0:
#             x = torch.cat(x, 1)
#             x = self.emb_drop(x)
#             x2 = self.bn1(x_cont)
#             x = torch.cat([x, x2], 1)
#         else:
#             x = self.bn1(x_cont.float())
#         x = F.relu(self.lin1(x))
#         x = self.drops(x)
#         x = self.bn2(x)
#         x = F.relu(self.lin2(x))
#         x = self.drops(x)
#         x = self.bn3(x)
#         x = self.lin3(x)
#         return x


class SeparableLinear(nn.Module):
    def __init__(self, in_num, out_num, drops, features):
        super().__init__()
        self.lin1 = nn.Linear(in_num, features)
        self.lin2 = nn.Linear(features, out_num)
        self.bn = nn.BatchNorm1d(features)
        self.drops = drops

    def forward(self, x):
        x = F.relu(self.lin1(x))
        x = self.drops(x)
        x = self.bn(x)
        return self.lin2(x)


class GeneralisedNeuralNetworkModel(nn.Module):
    def __init__(
        self,
        embedding_sizes,
        n_cont,
        n_class=1,
        dropout=0.5,
        emb_dropout=0.25,
        hidden_layers=1,
        features=150,
    ):
        super().__init__()
        self.embeddings = nn.ModuleList(
            [nn.Embedding(categories, size) for categories, size in embedding_sizes]
        )
        n_emb = sum(e.embedding_dim for e in self.embeddings)
        self.n_emb, self.n_cont = n_emb, n_cont
        self.drops = nn.Dropout(dropout)
        self.lin1 = nn.Linear(self.n_emb + self.n_cont, 200)
        self.lin2 = (
            nn.Linear(200, n_class)
            if hidden_layers <= 1
            else SeparableLinear(200, n_class, self.drops, features)
        )
        self.bn1 = nn.BatchNorm1d(self.n_cont)
        self.bn2 = nn.BatchNorm1d(200)
        self.emb_drop = nn.Dropout(emb_dropout)

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
