import torch
import torch.nn as nn
import numpy as np

# _lambda = 20
_gamma = 1


def get_default_device():
    if torch.cuda.is_available():
        return torch.device("cuda")
    else:
        return torch.device("cpu")


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
    loss = loss.squeeze()
    difficulty = difficulty.cpu().detach().numpy().squeeze()

    # loss = torch.mean(
    #     loss * torch.tensor(1 / _lambda ** (difficulty / epoch)).to(device)
    # )
    loss = torch.mean(loss * torch.tensor(difficulty ** (_gamma / epoch)).to(device))

    return loss


class BCECustomLoss(nn.Module):
    def __init__(self):
        super(BCECustomLoss, self).__init__()

    def forward(self, inputs, targets, difficulty, epoch):
        device = get_default_device()
        loss = -1 * (
            targets * torch.log(inputs) + (1 - targets) * torch.log(1 - inputs)
        )
        loss = torch.nan_to_num(loss)
        loss = torch.tensor(
            loss.cpu().detach().numpy() * (1 / _lambda ** (difficulty / epoch)),
            requires_grad=True,
        ).to(device)
        return loss.mean()
