from .data.airline_passenger_satisfaction_train import (
    preprocess_airline_data,
)
from .data.credit_card_fraud import preprocess_credit_card_ds
from .data.spotify_tracks_genre import (
    preprocess_spotify_tracks_ds,
)
from .data.stellar_ds import preprocess_stellar_ds
from .dataset_training import (
    train_nn_airline,
    test_nn_airline,
    train_nn_credit_card,
    test_nn_credit_card,
    train_nn_spotify_tracks,
    test_nn_spotify_tracks,
    train_nn_stellar,
    test_nn_stellar,
)
import torch
import numpy as np
import plac
import mlflow


@plac.opt("dataset", "Dataset to use for NN model evaluation", str, "d")
@plac.flg("relaxed", "Relax loss function flag", "r")
@plac.opt("batch_size", "Batch size of the data", int, "b")
@plac.opt("epochs", "Epochs number of training", int, "e")
@plac.opt("lr", "Learning rate of optimizer", float, "l")
@plac.flg("plot_map", "Flag whether to plot and save cartography map for dataset", "p")
@plac.opt(
    "rank_mode",
    "Mode in which the experiment is running: Cartography, Self-confidence, None",
    str,
    "m",
    ["cartography", "confidence", None],
)
def main(
    dataset: str = "credit_card",
    relaxed: bool = False,
    batch_size: int = 1000,
    epochs: int = 8,
    lr: float = 0.01,
    plot_map: bool = False,
    rank_mode: str = None,
):
    mlflow.set_experiment("methods_comparison_v2")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_tag(
            "mlflow.runName",
            f"{dataset}, rank_mode: {rank_mode}, relaxed: {relaxed}",
        )

        if dataset == "airline_passenger_satisfaction":
            train_df, test_df, embedded_cols = preprocess_airline_data()

            model = train_nn_airline(
                train_df,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                plot_map=plot_map,
            )
            test_acc = test_nn_airline(
                model, test_df, embedded_cols, batch_size=batch_size
            )
        elif dataset == "credit_card":
            X_rem, X_test, y_rem, y_test, prop = preprocess_credit_card_ds()

            model = train_nn_credit_card(
                X_rem,
                y_rem,
                prop,
                rank_mode,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                plot_map=plot_map,
            )
            test_acc = test_nn_credit_card(model, X_test, y_test, batch_size=batch_size)
        elif dataset == "spotify_tracks":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_spotify_tracks_ds()

            model = train_nn_spotify_tracks(
                X_rem,
                y_rem,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                plot_map=plot_map,
            )
            test_acc = test_nn_spotify_tracks(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )
        elif dataset == "stellar":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_stellar_ds()

            model = train_nn_stellar(
                X_rem,
                y_rem,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                plot_map=plot_map,
            )

            test_acc = test_nn_stellar(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )

        mlflow.log_param("dataset", dataset)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("relaxed", relaxed)
        mlflow.log_param("ranking_mode", rank_mode)

        mlflow.set_tag("Purpose", "Initial comparison")
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # main()
    plac.call(main)
