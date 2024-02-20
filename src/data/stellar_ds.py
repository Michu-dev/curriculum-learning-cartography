from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from imblearn.over_sampling import SMOTE
from collections import Counter

from dotenv import load_dotenv

load_dotenv()


def preprocess_stellar_ds() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
):
    ds_path = os.environ.get("PATH_TO_STELLAR_DS")
    star_df = pd.read_csv(ds_path, delimiter=",", header=0)
    star_df.drop(
        ["obj_ID", "run_ID", "rerun_ID", "cam_col", "field_ID", "spec_obj_ID"],
        axis=1,
        inplace=True,
    )

    numerical_cols = ["alpha", "delta", "u", "g", "r", "i", "z", "redshift", "MJD"]

    for col in star_df.columns:
        if col not in numerical_cols:
            star_df[col] = star_df[col].astype("category")

    embedded_cols = {
        n: len(col.cat.categories)
        for n, col in star_df.items()
        if col.name != "class"
        and col.dtype == "category"
        and len(col.cat.categories) > 2
    }
    x = star_df.drop(columns=["class"], axis=1)
    y = star_df.loc[:, "class"]

    smote = SMOTE(random_state=42)
    print("Original dataset shape %s" % Counter(y))
    x, y = smote.fit_resample(x, y)
    print("Resampled dataset shape %s" % Counter(y))

    star_df["class"] = LabelEncoder().fit_transform(star_df["class"])
    star_df["fiber_ID"] = LabelEncoder().fit_transform(star_df["fiber_ID"])
    star_df["plate"] = LabelEncoder().fit_transform(star_df["plate"])
    x = star_df.drop(columns=["class"], axis=1)
    y = star_df.loc[:, "class"]

    X_train, X_test, y_train, y_test = train_test_split(
        x, y, test_size=0.2, random_state=42, stratify=y
    )

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test, embedded_cols


class StellarDataset(Dataset):

    def __init__(
        self, X: pd.DataFrame, y: pd.DataFrame, embedded_col_names: dict, transform=None
    ) -> None:
        """
        Args:
            X: input_data
            y: labels
            embedded_col_names: list of categorical column names
            transform (callable, optional): Optional transform to be applied on a sample
        """
        self.X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)
        self.X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)
        self.y = y.copy().values.astype(np.int64)
        self.id = np.arange(len(self.y))
        self.difficulties = np.zeros(len(self.y), dtype=np.float32)

        self.transform = transform

    def __getitem__(self, index: int) -> tuple[np.int64, np.float32, np.float32]:
        sample = (
            self.id[index],
            self.X1[index],
            self.X2[index],
            self.y[index],
            self.difficulties[index],
        )

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.y)
