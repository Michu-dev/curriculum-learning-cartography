from typing import Any
import torch
import os
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import numpy as np

from pathlib import Path
from sklearn.preprocessing import LabelEncoder
from dotenv import load_dotenv

load_dotenv()


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=",", header=0)
    return df


def data_preprocessing(df: pd.DataFrame) -> pd.DataFrame:
    df = df.drop(["Unnamed: 0", "id"], axis=1)
    df["satisfaction"] = LabelEncoder().fit_transform(df["satisfaction"])
    df.columns = [c.replace(" ", "_") for c in df.columns]
    df["Arrival_Delay_in_Minutes"] = df["Arrival_Delay_in_Minutes"].fillna(
        df["Arrival_Delay_in_Minutes"].mean()
    )
    for col in df.columns:
        if df.dtypes[col] == "object":
            df[col] = LabelEncoder().fit_transform(df[col])
    return df


def preprocess_airline_data() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    train_path, test_path = os.environ.get("PATH_TO_AIRLINE_TRAIN"), os.environ.get(
        "PATH_TO_AIRLINE_TEST"
    )
    train_df, test_df = read_data(train_path), read_data(test_path)
    train_df, test_df = data_preprocessing(train_df), data_preprocessing(test_df)
    # TODO: try to join train and test df to preprocess embedded_cols
    numerical_cols = [
        "Age",
        "Flight_Distance",
        "Departure_Delay_in_Minutes",
        "Arrival_Delay_in_Minutes",
    ]
    for col in train_df.columns:
        if not col in numerical_cols:
            train_df[col] = train_df[col].astype("category")
            test_df[col] = test_df[col].astype("category")
    # print(train_df.dtypes)
    embedded_cols = {
        n: len(col.cat.categories)
        for n, col in train_df.items()
        if col.dtype == "category" and len(col.cat.categories) > 2
    }
    return train_df, test_df, embedded_cols


class AirlinePassengersDataset(Dataset):

    def __init__(self, X, y, embedded_col_names, transform=None) -> None:
        """
        Args:
            X: input data
            y: labels
            embedded_col_names: list of categorical column names
            transform (callable, optional): Optional transform to be applied on a sample
        """
        X = X.copy()
        self.X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        self.y = y.copy().values.astype(np.float32)

        self.transform = transform

    def __getitem__(self, index: int):
        sample = self.X1[index], self.X2[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.y)


class ToTensor:
    def __call__(self, sample):
        inputs, targets = sample
        return torch.from_numpy(inputs), torch.from_numpy(targets)


if __name__ == "__main__":
    print("")
# dataset = AirlinePassengersDataset(transform=None)
# first_data = dataset[0]

# features, labels = first_data
# print(features)
# print(len(dataset))
# print(type(features), type(labels))
