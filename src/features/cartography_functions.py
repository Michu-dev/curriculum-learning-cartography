from ..models.generalised_neural_network_model import GeneralisedNeuralNetworkModel
from ..train_run import DeviceDataLoader
import torch
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from pathlib import Path
from typing import List
import json
import logging
from tqdm import tqdm

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s - %(message)s", level=logging.INFO
)
logger = logging.getLogger(__name__)


def normalize_data(data: pd.Series) -> pd.Series:
    return (data - np.min(data)) / (np.max(data) - np.min(data))


def log_training_dynamics(
    output_dir: Path,
    epoch: int,
    train_ids: List[int],
    train_logits: List[List[float]],
    train_golds: List[int],
):
    """
    Save training dynamics (logits) from given epoch as records of a `.jsonl` file.
    """
    td_df = pd.DataFrame(
        {
            "guid": train_ids,
            f"logits_epoch_{epoch}": train_logits,
            "gold": train_golds,
        }
    )
    logging_dir = output_dir / "training_dynamics"
    if not logging_dir.exists():
        logging_dir.mkdir(parents=True, exist_ok=True)
    epoch_file_name = logging_dir / f"dynamics_epoch_{epoch}.jsonl"
    td_df.to_json(epoch_file_name, lines=True, orient="records")
    logger.info(f"Training Dynamics logged to {epoch_file_name}")


def plot_cartography_map(
    cartography_stats_df: pd.DataFrame,
    plot_dir: Path,
    title: str,
    show_hist: bool,
    model: str = "Feed-Forward-NN",
):
    # Set style.
    sns.set(style="whitegrid", font_scale=1.6, font="Georgia", context="paper")
    logger.info(f"Plotting figure for {title} using the Feed-Forward NN model ...")

    # Subsample data to plot, so the plot is not too busy.
    # cartography_stats_df = cartography_stats_df.sample(n=max_instances_to_plot if dataframe.shape[0] > max_instances_to_plot else len(dataframe))

    # Normalize correctness to a value between 0 and 1.
    cartography_stats_df = cartography_stats_df.assign(
        corr_frac=lambda d: d.correctness / d.correctness.max()
    )
    cartography_stats_df["correct."] = [
        round(x, 1) for x in cartography_stats_df["corr_frac"]
    ]

    main_metric = "variability"
    other_metric = "confidence"
    hue = "correct."

    num_hues = len(cartography_stats_df[hue].unique().tolist())
    style = hue

    if not show_hist:
        fig, ax0 = plt.subplots(1, 1, figsize=(8, 6))
    else:
        fig = plt.figure(
            figsize=(14, 10),
        )
        gs = fig.add_gridspec(3, 2, width_ratios=[5, 1])
        ax0 = fig.add_subplot(gs[:, 0])

    pal = sns.diverging_palette(260, 15, n=num_hues, sep=10, center="dark")

    plot = sns.scatterplot(
        x=main_metric,
        y=other_metric,
        ax=ax0,
        data=cartography_stats_df,
        hue=hue,
        palette=pal,
        style=style,
        s=30,
    )

    # Annotate Regions.
    bb = lambda c: dict(boxstyle="round,pad=0.3", ec=c, lw=2, fc="white")
    func_annotate = lambda text, xyc, bbc: ax0.annotate(
        text,
        xy=xyc,
        xycoords="axes fraction",
        fontsize=15,
        color="black",
        va="center",
        ha="center",
        rotation=350,
        bbox=bb(bbc),
    )
    an1 = func_annotate("ambiguous", xyc=(0.9, 0.5), bbc="black")
    an2 = func_annotate("easy-to-learn", xyc=(0.27, 0.85), bbc="r")
    an3 = func_annotate("hard-to-learn", xyc=(0.35, 0.25), bbc="b")

    if not show_hist:
        plot.legend(ncol=1, bbox_to_anchor=[0.175, 0.5], loc="right")
    else:
        plot.legend(fancybox=True, shadow=True, ncol=1)
    plot.set_xlabel("variability")
    plot.set_ylabel("confidence")

    if show_hist:
        plot.set_title(f"{title}-{model} Data Map", fontsize=17)

        # Make the histograms.
        ax1 = fig.add_subplot(gs[0, 1])
        ax2 = fig.add_subplot(gs[1, 1])
        ax3 = fig.add_subplot(gs[2, 1])

        plott0 = cartography_stats_df.hist(
            column=["confidence"], ax=ax1, color="#622a87"
        )
        plott0[0].set_title("")
        plott0[0].set_xlabel("confidence")
        plott0[0].set_ylabel("density")

        plott1 = cartography_stats_df.hist(column=["variability"], ax=ax2, color="teal")
        plott1[0].set_title("")
        plott1[0].set_xlabel("variability")
        plott1[0].set_ylabel("density")

        plot2 = sns.countplot(
            x="correct.", data=cartography_stats_df, ax=ax3, color="#86bf91"
        )
        ax3.xaxis.grid(True)  # Show the vertical gridlines

        plot2.set_title("")
        plot2.set_xlabel("correctness")
        plot2.set_ylabel("density")

    fig.tight_layout()
    filename = (
        f"{plot_dir}/{title}_{model}.pdf"
        if show_hist
        else f"{plot_dir}/compact_{title}_{model}.pdf"
    )
    fig.savefig(filename, dpi=300)
    logger.info(f"Plot saved to {filename}")


def read_training_dynamics(model_dir: Path) -> dict:
    train_dynamics = {}
    td_dir = model_dir / "training_dynamics"

    epochs = len([f for f in td_dir.iterdir() if f.is_file()])
    logger.info(f"Reading {epochs} files from {td_dir} ...")

    for epoch in tqdm(range(1, epochs + 1)):
        epoch_file = td_dir / f"dynamics_epoch_{epoch}.jsonl"
        with epoch_file.open() as md_file:
            content = md_file.readlines()
            for line in content:
                record = json.loads(line.strip())
                guid = record["guid"]
                if guid not in train_dynamics:
                    train_dynamics[guid] = {"gold": record["gold"], "logits": []}
                train_dynamics[guid]["logits"].append(record[f"logits_epoch_{epoch}"])

    logger.info(f"Read training dynamics for {len(train_dynamics)} train instances.")
    return train_dynamics


def data_cartography(
    model: GeneralisedNeuralNetworkModel,
    train_dl: DeviceDataLoader,
    loss_fn,
    optim,
    epochs: int,
    path_to_save_training_dynamics: Path,
    binary: bool = False,
) -> pd.DataFrame:

    if not path_to_save_training_dynamics.exists():
        path_to_save_training_dynamics.mkdir(parents=True, exist_ok=True)
        model.train()

        # think about adding eval set
        for epoch in range(epochs):
            total, sum_loss = 0, 0
            train_ids, train_golds, train_logits = None, None, None
            with tqdm(train_dl, unit="batch") as tepoch:
                for i, (idx, x1, x2, y, _) in enumerate(tepoch):
                    tepoch.set_description(f"Epoch {epoch+1}")

                    if 1 in list(y.shape) and not binary:
                        y = y.squeeze(dim=1)

                    batch = y.shape[0]
                    output = model(x1, x2)
                    loss = loss_fn(output, y)
                    optim.zero_grad()
                    loss.backward()
                    optim.step()
                    total += batch
                    sum_loss += batch * (loss.item())

                    if train_logits is None:
                        train_ids = idx.detach().cpu().numpy()
                        train_logits = output.detach().cpu().numpy()
                        train_golds = y.detach().cpu().numpy()
                    else:
                        train_ids = np.append(train_ids, idx.detach().cpu().numpy())
                        train_logits = np.append(
                            train_logits, output.detach().cpu().numpy(), axis=0
                        )
                        train_golds = np.append(train_golds, y.detach().cpu().numpy())

                    tepoch.set_postfix(loss=sum_loss / total)

            log_training_dynamics(
                path_to_save_training_dynamics,
                epoch + 1,
                list(train_ids),
                list(train_logits),
                list(train_golds),
            )

    training_dynamics = read_training_dynamics(path_to_save_training_dynamics)
    confidence_ = {}
    variability_ = {}
    correctness_ = {}
    cartography_dir = path_to_save_training_dynamics / "td_metrics"
    td_metrics_filename = cartography_dir / "td_metrics.jsonl"

    if not cartography_dir.exists():
        cartography_dir.mkdir(parents=True, exist_ok=True)
        logger.info(
            f"Compute training dynamic metrics for {len(training_dynamics)} train instances."
        )

        for guid in tqdm(training_dynamics):
            correctness_trend = []
            true_probs_trend = []

            record = training_dynamics[guid]
            for epoch_logits in record["logits"]:
                probs = (
                    torch.nn.functional.sigmoid(torch.Tensor(epoch_logits))
                    if binary
                    else torch.nn.functional.softmax(torch.Tensor(epoch_logits), dim=-1)
                )
                probs = torch.Tensor([1.0 - probs[0], probs[0]]) if binary else probs
                true_class_prob = float(probs[int(record["gold"])])
                true_probs_trend.append(true_class_prob)

                prediction = np.argmax(probs)
                is_correct = (prediction == record["gold"]).item()
                correctness_trend.append(is_correct)

            correctness_[guid] = sum(correctness_trend)
            confidence_[guid] = np.mean(true_probs_trend)
            variability_[guid] = np.std(true_probs_trend)

        column_names = [
            "guid",
            "confidence",
            "variability",
            "correctness",
        ]
        df = pd.DataFrame(
            [
                [
                    guid,
                    confidence_[guid],
                    variability_[guid],
                    correctness_[guid],
                ]
                for guid in correctness_
            ],
            columns=column_names,
        )
        df.to_json(td_metrics_filename, orient="records", lines=True)
    else:
        logger.info(f"Read training dynamic metric.")
        df = pd.read_json(td_metrics_filename, orient="records", lines=True)

    return df
