import torch
import torch.nn as nn
import numpy as np
from .auxiliary_functions import get_default_device

_lambda = 20


def relax_loss(loss: torch.Tensor, difficulty: np.ndarray, epoch: int):
    device = get_default_device()

    # print(loss[:5])
    # print("-----------------------------------")
    # print(difficulty[:5])

    # print("-----------------------------------")

    # print(
    #     torch.tensor(
    #         loss.cpu().detach().numpy() * (1 / _lambda ** (difficulty / epoch)),
    #         requires_grad=True,
    #     ).to(device)[:5]
    # )

    return torch.mean(
        torch.tensor(
            loss.cpu().detach().numpy() * (1 / _lambda ** (difficulty / epoch)),
            requires_grad=True,
        ).to(device)
    )


class BCECustomLoss(nn.Module):
    def __init__(self):
        super(BCECustomLoss, self).__init__()

    def forward(self, inputs, targets, difficulty, epoch):
        device = None
        if torch.cuda.is_available():
            device = torch.device("cuda")
        else:
            device = torch.device("cpu")
        loss = -1 * (
            targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)
        )
        loss = torch.nan_to_num(loss)
        loss = torch.tensor(
            loss.cpu().detach().numpy() * (1 / _lambda ** (difficulty / epoch)),
            requires_grad=True,
        ).to(device)
        return loss.mean()
