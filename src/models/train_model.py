from ..data.airline_passenger_satisfaction_train import (
    preprocess_airline_data,
)
from ..data.credit_card_fraud import preprocess_credit_card_ds
from ..data.spotify_tracks_genre import (
    preprocess_spotify_tracks_ds,
)
from ..data.stellar_ds import preprocess_stellar_ds
import torch
from .generalised_neural_network_model import GeneralisedNeuralNetworkModel
from .training_functions import (
    train_nn_airline,
    test_nn_airline,
    train_nn_credit_card,
    test_nn_credit_card,
    train_nn_spotify_tracks,
    test_nn_spotify_tracks,
    train_nn_stellar,
    test_nn_stellar,
)

# from data.airline_passenger_satisfaction_train import read_data, data_preprocessing, AirlinePassengersDataset, ToTensor
import pandas as pd
import numpy as np
import plac
import mlflow
from skorch import NeuralNetClassifier
from imblearn.over_sampling import SMOTE


@plac.opt("dataset", "Dataset to use for NN model evaluation", str, "d")
@plac.opt("relaxed", "Relax loss function flag", bool, "r")
@plac.opt("batch_size", "Batch size of the data", int, "b")
@plac.opt("epochs", "Epochs number of training", int, "e")
@plac.opt("lr", "Learning rate of optimizer", float, "l")
def main(
    dataset: str = "credit_card",
    relaxed: bool = False,
    batch_size: int = 1000,
    epochs: int = 8,
    lr: float = 0.01,
):
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")

        if dataset == "airline_passenger_satisfaction":
            train_df, test_df, embedded_cols = preprocess_airline_data()
            full_df = pd.concat([train_df, test_df], axis=0)
            embedded_col_names = embedded_cols.keys()
            X, y = full_df.iloc[:, :-1], full_df.iloc[:, [-1]]
            X1 = X.loc[:, embedded_col_names].copy().values.astype(np.int64)
            X2 = X.drop(columns=embedded_col_names).copy().values.astype(np.float32)
            y = y.copy().values.astype(np.float32)

            X = {
                "x_cat": X1,
                "x_cont": X2,
            }

            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]
            skorch_model = NeuralNetClassifier(
                GeneralisedNeuralNetworkModel(embedding_sizes, 7),
                criterion=torch.nn.BCEWithLogitsLoss,
                max_epochs=8,
            )

            if relaxed:
                skorch_model.fit(X, y)

            model = train_nn_airline(
                train_df,
                embedded_cols,
                skorch_model,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            final_probs = test_nn_airline(
                model, test_df, embedded_cols, batch_size=batch_size
            )
        elif dataset == "credit_card":
            X_rem, X_test, y_rem, y_test, prop = preprocess_credit_card_ds()
            # Apply SMOTE oversampling to the training data
            smote = SMOTE(random_state=42)
            X_sampled, y_sampled = smote.fit_resample(X_rem, y_rem)

            X = X_sampled.copy().astype(np.float32)
            y = y_sampled.copy().values.astype(np.float32)

            X = {
                "x_cat": np.zeros(X.shape[0], dtype=np.float32),
                "x_cont": X,
            }

            skorch_model = NeuralNetClassifier(
                GeneralisedNeuralNetworkModel([], len(X_rem[0])),
                criterion=torch.nn.BCEWithLogitsLoss,
                max_epochs=8,
            )

            if relaxed:
                skorch_model.fit(X, y)

            model = train_nn_credit_card(
                X_rem,
                y_rem,
                prop,
                skorch_model,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            test_acc = test_nn_credit_card(model, X_test, y_test, batch_size=batch_size)
        elif dataset == "spotify_tracks":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_spotify_tracks_ds()
            embedded_col_names = embedded_cols.keys()

            X1 = X_rem.loc[:, embedded_col_names].copy().values.astype(np.int64)
            X2 = X_rem.drop(columns=embedded_col_names).copy().values.astype(np.float32)
            y = y_rem.copy().values.astype(np.int64).squeeze()

            X = {
                "x_cat": X1,
                "x_cont": X2,
            }
            n_cont = len(X_rem.columns) - len(embedded_cols)

            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]
            skorch_model = NeuralNetClassifier(
                GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=114),
                criterion=torch.nn.CrossEntropyLoss,
                max_epochs=8,
            )

            if relaxed:
                skorch_model.fit(X, y)

            model = train_nn_spotify_tracks(
                X_rem,
                y_rem,
                embedded_cols,
                skorch_model,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )
            test_acc = test_nn_spotify_tracks(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )
        elif dataset == "stellar":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_stellar_ds()
            embedded_col_names = embedded_cols.keys()

            X1 = X_rem.loc[:, embedded_col_names].copy().values.astype(np.int64)
            X2 = X_rem.drop(columns=embedded_col_names).copy().values.astype(np.float32)
            y = y_rem.copy().values.astype(np.int64).squeeze()
            X = {
                "x_cat": X1,
                "x_cont": X2,
            }
            n_cont = len(X_rem.columns) - len(embedded_cols)
            embedding_sizes = [
                (n_categories + 1, min(50, (n_categories + 1) // 2))
                for _, n_categories in embedded_cols.items()
            ]

            skorch_model = NeuralNetClassifier(
                GeneralisedNeuralNetworkModel(embedding_sizes, n_cont, n_class=3),
                criterion=torch.nn.CrossEntropyLoss,
                max_epochs=8,
            )

            if relaxed:
                skorch_model.fit(X, y)

            model = train_nn_stellar(
                X_rem,
                y_rem,
                embedded_cols,
                skorch_model,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
            )

            test_acc = test_nn_stellar(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )

        mlflow.log_param("dataset", dataset)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("relaxed", relaxed)

        mlflow.set_tag("Purpose", "Initial comparison")
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # main()
    plac.call(main)
