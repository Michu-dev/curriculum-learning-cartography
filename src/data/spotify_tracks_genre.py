from torch.utils.data import Dataset
import os
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler

from dotenv import load_dotenv

load_dotenv()


def preprocess_spotify_tracks_ds() -> (
    tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict]
):
    ds_path = os.environ.get("PATH_TO_SPOTIFY_GENRE")
    spotify_tracks_df = pd.read_csv(ds_path, delimiter=",", header=0)
    spotify_tracks_df = spotify_tracks_df.dropna()
    spotify_tracks_df.drop(
        ["Unnamed: 0", "track_id", "track_name", "album_name", "artists"],
        axis=1,
        inplace=True,
    )
    spotify_tracks_df.rename(columns={"track_genre": "genre"}, inplace=True)
    spotify_tracks_df["genre"] = LabelEncoder().fit_transform(
        spotify_tracks_df["genre"]
    )
    spotify_tracks_df["explicit"] = spotify_tracks_df["explicit"].astype(int)

    numerical_cols = [
        "duration_ms",
        "danceability",
        "energy",
        "loudness",
        "speechiness",
        "acousticness",
        "instrumentalness",
        "liveness",
        "valence",
        "tempo",
    ]
    for col in spotify_tracks_df.columns:
        if col not in numerical_cols:
            spotify_tracks_df[col] = spotify_tracks_df[col].astype("category")

    embedded_cols = {
        n: len(col.cat.categories)
        for n, col in spotify_tracks_df.items()
        if col.dtype == "category" and len(col.cat.categories) > 2
    }
    del embedded_cols["genre"]
    X_train, X_test, y_train, y_test = train_test_split(
        spotify_tracks_df.drop(columns=["genre"]),
        spotify_tracks_df.loc[:, ["genre"]],
        test_size=0.2,
        random_state=42,
    )

    scaler = StandardScaler()
    X_train[numerical_cols] = scaler.fit_transform(X_train[numerical_cols])
    X_test[numerical_cols] = scaler.transform(X_test[numerical_cols])

    return X_train, X_test, y_train, y_test, embedded_cols


class SpotifyTracksDataset(Dataset):

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

        self.transform = transform

    def __getitem__(self, index: int) -> tuple[np.int64, np.float32, np.float32]:
        sample = self.id[index], self.X1[index], self.X2[index], self.y[index]

        if self.transform:
            sample = self.transform(sample)

        return sample

    def __len__(self) -> int:
        return len(self.y)
