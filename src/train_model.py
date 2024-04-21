from .data.airline_passenger_satisfaction_train import (
    preprocess_airline_data,
)
from .data.credit_card_fraud import preprocess_credit_card_ds
from .data.spotify_tracks_genre import (
    preprocess_spotify_tracks_ds,
)
from .data.stellar_ds import preprocess_stellar_ds
from .data.fashion_mnist import get_fashion_mnist_data
from .dataset_training import (
    train_nn_airline,
    test_nn_airline,
    train_nn_credit_card,
    test_nn_credit_card,
    train_nn_spotify_tracks,
    test_nn_spotify_tracks,
    train_nn_stellar,
    test_nn_stellar,
    train_cnn_fashion_mnist,
    test_cnn_fashion_mnist,
)
import torch
import numpy as np
import plac
import mlflow


@plac.opt("dataset", "Dataset to use for NN model evaluation", str, "d")
@plac.flg("relaxed", "Relax loss function flag", "r")
@plac.flg(
    "ranked", "Rank examples from training set in level of difficulty flag", "rnk"
)
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
@plac.opt("alpha", "Alpha parameter of 2D -> 1D mapping in Cartography", float, "a")
@plac.opt("beta", "Beta parameter of 2D -> 1D mapping in Cartography", float, "bt")
@plac.opt("gamma", "Gamma parameter of relaxation loss function", float, "g")
def main(
    dataset: str = "credit_card",
    relaxed: bool = False,
    ranked: bool = False,
    batch_size: int = 1000,
    epochs: int = 8,
    lr: float = 0.01,
    plot_map: bool = False,
    rank_mode: str = None,
    alpha: float = None,
    beta: float = None,
    gamma: float = 2.0,
):
    mlflow.set_experiment("fashion_mnist_cnn")
    torch.manual_seed(0)
    np.random.seed(0)
    torch.use_deterministic_algorithms(True)
    with mlflow.start_run():
        mlflow.set_tracking_uri("http://127.0.0.1:5000")
        mlflow.set_tag(
            "mlflow.runName",
            f"{dataset}, rank_mode: {rank_mode}, relaxed: {relaxed}, ranked: {ranked}",
        )

        if dataset == "airline_passenger_satisfaction":
            train_df, test_df, embedded_cols = preprocess_airline_data()

            # Params from NAS/HT
            batch_size = 32
            lr = 0.000355
            weight_decay = 0.03598
            optimizer = "Adam"
            hidden_layers = 2
            dropout = 0.2973
            emb_dropout = 0.2002
            features = 361

            model = train_nn_airline(
                train_df,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                ranked=ranked,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                features=features,
                plot_map=plot_map,
                wd=weight_decay,
            )
            test_acc = test_nn_airline(
                model, test_df, embedded_cols, batch_size=batch_size
            )
        elif dataset == "credit_card":
            X_rem, X_test, y_rem, y_test, prop = preprocess_credit_card_ds()

            # Params from NAS/HT
            batch_size = 2048
            lr = 0.04255
            weight_decay = 0.06703
            optimizer = "Adam"
            hidden_layers = 2
            dropout = 0.43460
            emb_dropout = 0.35049
            features = 207

            model = train_nn_credit_card(
                X_rem,
                y_rem,
                prop,
                rank_mode,
                relaxed=relaxed,
                ranked=ranked,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                features=features,
                plot_map=plot_map,
                wd=weight_decay,
            )
            test_acc = test_nn_credit_card(model, X_test, y_test, batch_size=batch_size)
        elif dataset == "spotify_tracks":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_spotify_tracks_ds()

            # Params from NAS/HT
            batch_size = 64
            lr = 0.000654
            weight_decay = 0.00257
            optimizer = "Adam"
            hidden_layers = 1
            dropout = 0.24486
            emb_dropout = 0.56629
            features = 397

            model = train_nn_spotify_tracks(
                X_rem,
                y_rem,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                ranked=ranked,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                features=features,
                plot_map=plot_map,
                wd=weight_decay,
            )
            test_acc = test_nn_spotify_tracks(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )
        elif dataset == "stellar":
            X_rem, X_test, y_rem, y_test, embedded_cols = preprocess_stellar_ds()

            # Params from NAS/HT
            batch_size = 128
            lr = 0.000368
            weight_decay = 0.0553
            optimizer = "Adam"
            hidden_layers = 2
            dropout = 0.11415
            emb_dropout = 0.38323
            features = 271

            model = train_nn_stellar(
                X_rem,
                y_rem,
                embedded_cols,
                rank_mode,
                relaxed=relaxed,
                ranked=ranked,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                hidden_layers=hidden_layers,
                dropout=dropout,
                emb_dropout=emb_dropout,
                features=features,
                plot_map=plot_map,
                wd=weight_decay,
            )

            test_acc = test_nn_stellar(
                model, X_test, y_test, embedded_cols, batch_size=batch_size
            )
        elif dataset == "fashion_mnist":
            train_data, test_data = get_fashion_mnist_data()
            X_rem, y_rem = map(list, zip(*[[x[0].numpy(), x[1]] for x in train_data]))
            X_test, y_test = map(list, zip(*[[x[0].numpy(), x[1]] for x in test_data]))

            # Params from NAS/HT
            # conv_layers, dense_layers = 2, 2
            batch_size = 128
            lr = 0.01933
            weight_decay = 0.02574
            optimizer = "SGD"

            model = train_cnn_fashion_mnist(
                X_rem,
                y_rem,
                rank_mode,
                relaxed=relaxed,
                ranked=ranked,
                alpha=alpha,
                beta=beta,
                gamma=gamma,
                epochs=epochs,
                batch_size=batch_size,
                lr=lr,
                optimizer=optimizer,
                plot_map=plot_map,
                wd=weight_decay,
            )

            test_acc = test_cnn_fashion_mnist(
                model, X_test, y_test, batch_size=batch_size
            )

        mlflow.log_param("dataset", dataset)
        mlflow.log_param("batch_size", batch_size)
        mlflow.log_param("epochs", epochs)
        mlflow.log_param("learning_rate", lr)
        mlflow.log_param("relaxed", relaxed)
        mlflow.log_param("ranked", ranked)
        mlflow.log_param("ranking_mode", rank_mode)
        mlflow.log_param("alpha", alpha)
        mlflow.log_param("beta", beta)
        mlflow.log_param("gamma", gamma)

        mlflow.set_tag("Purpose", "Initial comparison")
        mlflow.pytorch.log_model(model, "model")


if __name__ == "__main__":
    # main()
    plac.call(main)
