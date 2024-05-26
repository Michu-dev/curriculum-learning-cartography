import mlflow
import pandas as pd
import warnings
from mlflow.tracking import MlflowClient
from tqdm import tqdm
import matplotlib.pyplot as plt
import numpy as np
import argparse
import scipy.stats as stats


def setup_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Sample argument parser")
    parser.add_argument(
        "-d",
        action="store",
        choices=[
            "airline_passenger_satisfaction",
            "credit_card",
            "stellar",
            "fashion_mnist",
        ],
        default="stellar",
        help="set the dataset to plot metrics history or conduct independent t-test",
    )
    parser.add_argument(
        "-m",
        action="store",
        choices=["train_loss", "valid_loss", "valid_accuracy", "test_acc", "f1_score"],
        default="train_loss",
        help="set the metric to plot or conduct independent t-test",
    )

    parser.add_argument(
        "-t",
        action="store_true",
        help="set script to run independent t-test for a specified combination of 2 methods",
    )

    parser.add_argument(
        "-rm1",
        action="store",
        choices=["None", "cartography", "confidence"],
        default="None",
        help="set ranking_mode for independent t-test - probe 1",
    )

    parser.add_argument(
        "-rm2",
        action="store",
        choices=["None", "cartography", "confidence"],
        default="None",
        help="set ranking_mode for independent t-test - probe 2",
    )

    parser.add_argument(
        "-rnk1",
        action="store_true",
        help="set ranked boolean for independent t-test - probe 1",
    )

    parser.add_argument(
        "-rnk2",
        action="store_true",
        help="set ranked boolean for independent t-test - probe 2",
    )

    parser.add_argument(
        "-rl1",
        action="store_true",
        help="set relaxed boolean for independent t-test - probe 1",
    )

    parser.add_argument(
        "-rl2",
        action="store_true",
        help="set relaxed boolean for independent t-test - probe 2",
    )

    parser.add_argument(
        "-g1",
        type=float,
        choices=[1.5, 2.0, 2.5, 5.0],
        help="set gamma value for independent t-test - probe 1",
    )

    parser.add_argument(
        "-g2",
        type=float,
        choices=[1.5, 2.0, 2.5, 5.0],
        help="set gamma value for independent t-test - probe 2",
    )

    return parser


def read_experiments(experiment_names: list) -> pd.DataFrame:
    all_runs = mlflow.search_runs(experiment_names=experiment_names)
    print(all_runs.columns)
    cols = [
        "params.dataset",
        "params.ranking_mode",
        "params.ranked",
        "params.relaxed",
        "params.gamma",
        "metrics.test_acc",
        "metrics.f1_score",
        # "metrics.roc_auc",
        "metrics.train_loss",
        "metrics.validation_loss",
        "metrics.Accuracy",
    ]
    experiments_df = all_runs[cols]
    cols_dict = {c: c[c.find(".") + 1 :] for c in cols}
    experiments_df.rename(columns=cols_dict, inplace=True)
    experiments_df["gamma"] = experiments_df["gamma"].astype(float)
    bool_conv_dict = {"True": True, "False": False}
    experiments_df["ranked"].replace(bool_conv_dict, inplace=True)
    experiments_df["relaxed"].replace(bool_conv_dict, inplace=True)
    return experiments_df


def get_methods_df(experiments_df: pd.DataFrame, with_agg: bool = True) -> pd.DataFrame:
    results_df = experiments_df[
        ~((experiments_df["relaxed"] == True) & (experiments_df["gamma"] >= 2.0))
    ]
    if with_agg:
        grouping_cols = ["dataset", "ranking_mode", "ranked", "relaxed"]
        agg_functions = {
            "test_acc": ["mean", "std"],
            "f1_score": ["mean", "std"],
            "train_loss": ["mean", "std"],
        }
        results_df = results_df.groupby(grouping_cols).agg(agg_functions)
    return results_df


def get_gammas_df(experiments_df: pd.DataFrame, with_agg: bool = True) -> pd.DataFrame:
    results_df = experiments_df[experiments_df["relaxed"] == True]
    if with_agg:
        grouping_cols = ["dataset", "ranking_mode", "ranked", "gamma"]
        agg_functions = {
            "test_acc": ["mean", "std"],
            "f1_score": ["mean", "std"],
            "train_loss": ["mean", "std"],
        }
        results_df = results_df.groupby(grouping_cols).agg(agg_functions)
    return results_df


def get_plots_data(
    experiments_df: pd.DataFrame, comparison_type: str = "methods"
) -> pd.DataFrame:
    if comparison_type == "methods":
        experiments_df = get_methods_df(experiments_df, with_agg=False)
        grouping_cols = ["dataset", "ranking_mode", "ranked", "relaxed"]
    else:
        experiments_df = get_gammas_df(experiments_df, with_agg=False)
        grouping_cols = ["dataset", "ranking_mode", "ranked", "gamma"]

    dict_len = len(eval(experiments_df["train_loss"].iloc[0]))
    experiments_df["train_loss"] = experiments_df["train_loss"].map(eval)
    experiments_df["valid_loss"] = experiments_df["valid_loss"].map(eval)
    experiments_df["valid_accuracy"] = experiments_df["valid_accuracy"].map(eval)
    experiments_df = pd.concat(
        [
            experiments_df.drop(["train_loss"], axis=1),
            experiments_df["train_loss"].apply(pd.Series).add_prefix("train_loss_"),
        ],
        axis=1,
    )
    experiments_df = pd.concat(
        [
            experiments_df.drop(["valid_loss"], axis=1),
            experiments_df["valid_loss"].apply(pd.Series).add_prefix("valid_loss_"),
        ],
        axis=1,
    )
    experiments_df = pd.concat(
        [
            experiments_df.drop(["valid_accuracy"], axis=1),
            experiments_df["valid_accuracy"]
            .apply(pd.Series)
            .add_prefix("valid_accuracy_"),
        ],
        axis=1,
    )
    agg_functions = dict()
    for metric in ("train_loss", "valid_loss", "valid_accuracy"):
        for i in range(dict_len):
            agg_functions[metric + "_" + str(i)] = ["mean", "std"]
    results_df = experiments_df.groupby(grouping_cols).agg(agg_functions)
    return results_df


def plot_training_progress(
    agg_df: pd.DataFrame, dataset: str, metric: str, comparison_type: str = "methods"
):
    ds_agg_df = agg_df[agg_df.index.get_level_values(0) == dataset]
    x = list(range(1, 9))

    # fig, axs = plt.subplots(3)
    if comparison_type == "methods":
        variants = [
            ("None", False, False),
            ("cartography", False, True),
            ("cartography", True, False),
            ("cartography", True, True),
            ("confidence", False, True),
            ("confidence", True, False),
            ("confidence", True, True),
        ]
    else:
        variants = [
            ("cartography", False, 1.5),
            ("cartography", False, 2),
            ("cartography", False, 2.5),
            ("cartography", False, 5),
            ("cartography", True, 1.5),
            ("cartography", True, 2),
            ("cartography", True, 2.5),
            ("cartography", True, 5),
            ("confidence", False, 1.5),
            ("confidence", False, 2),
            ("confidence", False, 2.5),
            ("confidence", False, 5),
            ("confidence", True, 1.5),
            ("confidence", True, 2),
            ("confidence", True, 2.5),
            ("confidence", True, 5),
        ]

        # metrics = ("train_loss", "valid_loss", "valid_accuracy")

    relaxed_gamma_tag = "relaxed" if comparison_type == "methods" else "gamma"
    for ranking_mode, ranked, relaxed_or_gamma in variants:
        means, stds = [], []
        for i in range(8):
            mean = ds_agg_df[
                (ds_agg_df.index.get_level_values(2) == ranked)
                & (ds_agg_df.index.get_level_values(3) == relaxed_or_gamma)
                & (ds_agg_df.index.get_level_values(1) == ranking_mode)
            ][(f"{metric}_{i}", "mean")]
            std = ds_agg_df[
                (ds_agg_df.index.get_level_values(2) == ranked)
                & (ds_agg_df.index.get_level_values(3) == relaxed_or_gamma)
                & (ds_agg_df.index.get_level_values(1) == ranking_mode)
            ][(f"{metric}_{i}", "std")]
            means.append(float(mean))
            stds.append(float(std))

        plt.plot(
            x,
            means,
            label=f"ranking_mode: {ranking_mode}, ranked: {ranked}, {relaxed_gamma_tag}: {relaxed_or_gamma}",
        )
        plt.fill_between(x, np.subtract(means, stds), np.add(means, stds), alpha=0.2)

    # for n, metric in enumerate(metrics):
    # if n == 0:
    #     axs[n].set_title(f"Training metrics plot for {dataset}")
    # # axs[n].set_title(f"Metric: {metric} plot")
    # axs[n].set_xlabel("Epoch number")
    # axs[n].set_ylabel(metric)

    plt.title(f"Training {metric} metric plot for {dataset}", y=1.08)
    plt.xlabel("Epoch number")
    plt.ylabel(metric)
    plt.legend(
        loc="upper center",
        bbox_to_anchor=(0.5, 1.05),
        ncol=3,
        fancybox=True,
        shadow=True,
    )

    # plt.tight_layout()
    plt.show()


def read_training_metrics(experiment_names: list) -> pd.DataFrame:
    all_runs = mlflow.search_runs(experiment_names=experiment_names)
    print(all_runs.columns)
    cols = [
        "run_id",
        "params.dataset",
        "params.ranking_mode",
        "params.ranked",
        "params.relaxed",
        "params.gamma",
    ]
    experiments_df = all_runs[cols]
    cols_dict = {c: c[c.find(".") + 1 :] for c in cols}
    experiments_df.rename(columns=cols_dict, inplace=True)
    run_ids = experiments_df["run_id"].to_numpy()
    client = MlflowClient(tracking_uri="http://127.0.0.1:5000")
    train_loss_history, valid_loss_history, valid_accuracy_history = [], [], []
    for run_id in tqdm(run_ids):
        train_loss_metric = client.get_metric_history(run_id, "train_loss")
        train_loss_metric = {
            train_loss.step: train_loss.value for train_loss in train_loss_metric
        }
        valid_loss_metric = client.get_metric_history(run_id, "validation_loss")
        valid_loss_metric = {
            valid_loss.step: valid_loss.value for valid_loss in valid_loss_metric
        }
        valid_accuracy_metric = client.get_metric_history(run_id, "Accuracy")
        valid_accuracy_metric = {
            valid_accuracy.step: valid_accuracy.value
            for valid_accuracy in valid_accuracy_metric
        }

        train_loss_history.append(train_loss_metric)
        valid_loss_history.append(valid_loss_metric)
        valid_accuracy_history.append(valid_accuracy_metric)

    experiments_df["train_loss"] = train_loss_history
    experiments_df["valid_loss"] = valid_loss_history
    experiments_df["valid_accuracy"] = valid_accuracy_history

    return experiments_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = setup_parser()

    args = parser.parse_args()
    dataset, metric, test_mode = args.d, args.m, args.t
    ranking_mode_1, ranking_mode_2 = args.rm1, args.rm2
    ranked_1, ranked_2 = args.rnk1, args.rnk2
    relaxed_1, relaxed_2 = args.rl1, args.rl2
    gamma_1, gamma_2 = args.g1, args.g2

    experiment_names = [
        "airline_ds_experiments",
        "credit_card_ds_experiments",
        "stellar_ds_experiments",
        "fashion_mnist_ds_experiments",
    ]

    if test_mode:
        experiments_df = read_experiments(experiment_names)

        experiments_df = get_methods_df(experiments_df, with_agg=False)
        # experiments_df = get_gammas_df(experiments_df, with_agg=False)

        print(
            f"First method: ranked: {ranked_1}, relaxed: {relaxed_1}, ranking_mode: {ranking_mode_1}"
        )
        print(
            f"Second method: ranked: {ranked_2}, relaxed: {relaxed_2}, ranking_mode: {ranking_mode_2}"
        )
        probe1 = experiments_df[
            (experiments_df["dataset"] == dataset)
            & (experiments_df["ranked"] == ranked_1)
            & (experiments_df["relaxed"] == relaxed_1)
            & (experiments_df["ranking_mode"] == ranking_mode_1)
        ][metric].to_numpy()
        probe2 = experiments_df[
            (experiments_df["dataset"] == dataset)
            & (experiments_df["ranked"] == ranked_2)
            & (experiments_df["relaxed"] == relaxed_2)
            & (experiments_df["ranking_mode"] == ranking_mode_2)
        ][metric].to_numpy()

        print(probe1[:3], probe2[:3])
        print(len(probe1), len(probe2))
        print(stats.ttest_ind(probe1, probe2))
        p_value = stats.ttest_ind(probe1, probe2).pvalue
        if p_value <= 0.05:
            print("H0 REJECTION")

    else:
        # experiments_df = read_training_metrics(experiment_names)
        experiments_df = pd.read_csv("training_history.csv", index_col=0)
        experiments_df["ranking_mode"] = experiments_df["ranking_mode"].astype(str)
        experiments_df["ranking_mode"] = experiments_df["ranking_mode"].replace(
            {"nan": "None"}
        )
        methods_results_df = get_plots_data(experiments_df, comparison_type="methods")

        print(methods_results_df)
        plot_training_progress(
            methods_results_df, dataset, metric, comparison_type="methods"
        )

        # gammas_results_df = get_gammas_df(experiments_df)
        # print(gammas_results_df)

    # experiment_names_dict = {dict(mlflow.get_experiment_by_name(e))['experiment_id']: e for e in experiment_names}
    # all_runs["experiment_name"] = all_runs.apply(lambda r: experiment_names_dict[r.experiment_id], axis=1)
    # with pd.ExcelWriter("results.xlsx") as writer:
    #     methods_results_df.to_excel(writer, sheet_name="methods", index=True)
    #     gammas_results_df.to_excel(writer, sheet_name="gammas", index=True)
