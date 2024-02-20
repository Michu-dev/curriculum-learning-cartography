from torch.utils.data import Dataset
import os
from pathlib import Path
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

from dotenv import load_dotenv

load_dotenv()


def read_data(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, delimiter=",", header=0)
    return df


def preprocess_credit_card_ds() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, list]
):
    ds_path = os.environ.get("PATH_TO_CREDIT_CARD")
    credit_card_df = read_data(ds_path)
    proportions_list = (credit_card_df["Class"].value_counts()).tolist()
    proportions_list = [1.0 / p for p in proportions_list]

    scaler = StandardScaler()
    X = credit_card_df.drop(columns=["Time", "Class"])
    y = credit_card_df.loc[:, ["Class"]]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    X_train = scaler.fit_transform(X_train)
    X_test = scaler.transform(X_test)
    return X_train, X_test, y_train, y_test, proportions_list


class CreditCardDataset(Dataset):

    def __init__(self, X: pd.DataFrame, y: pd.DataFrame, transform=None):
        """
        Args:
            X: input_data
            y: labels
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.X = X.copy().astype(np.float32)
        self.y = y.copy().values.astype(np.float32)
        self.id = np.arange(len(self.y))
        self.transform = transform

    # Workaround with 0 returned always instead of embedded col values for categorical columns
    def __getitem__(self, index: int):
        sample = self.id[index], 0, self.X[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self):
        return len(self.y)
