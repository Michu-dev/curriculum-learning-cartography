import torch.nn as nn
import torch.nn.functional as F
import math


class FashionMNISTModel(nn.Module):
    def __init__(
        self,
        input_shape: int,
        hidden_units: int,
        output_shape: int,
        kernel_conv: int,
        padding_conv: int,
        stride_conv: int,
        kernel_pooling: int,
        padding_pooling: int,
        stride_pooling: int,
        dropout: float,
        features: int,
    ):
        super().__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(
                in_channels=input_shape,
                out_channels=hidden_units,
                kernel_size=kernel_conv,
                padding=padding_conv,
                stride=stride_conv,
            ),
            nn.BatchNorm2d(hidden_units),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=kernel_pooling,
                stride=stride_pooling,
                padding=padding_pooling,
            ),
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(
                in_channels=hidden_units,
                out_channels=hidden_units * 2,
                kernel_size=kernel_conv,
                padding=padding_conv,
                stride=stride_conv,
            ),
            nn.BatchNorm2d(hidden_units * 2),
            nn.ReLU(),
            nn.MaxPool2d(
                kernel_size=kernel_pooling,
                stride=stride_pooling,
                padding=padding_pooling,
            ),
        )
        self.out_size = (
            math.floor((28 - kernel_conv + 2 * padding_conv) / stride_conv) + 1
        )
        self.out_size = (
            math.floor(
                (self.out_size - kernel_pooling + 2 * padding_pooling) / stride_pooling
            )
            + 1
        )
        self.out_size = (
            math.floor((28 - kernel_conv + 2 * padding_conv) / stride_conv) + 1
        )
        self.out_size = (
            math.floor(
                (self.out_size - kernel_pooling + 2 * padding_pooling) / stride_pooling
            )
            + 1
        )
        self.conv = nn.Sequential(self.conv1, self.conv2)
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(
                in_features=3800,
                out_features=4 * features,
            ),
            nn.Dropout(dropout),
            nn.Linear(in_features=4 * features, out_features=features),
            nn.Linear(in_features=features, out_features=output_shape),
        )

    def forward(self, x, x1=None):
        x = self.conv(x)
        x = self.classifier(x)
        return x
