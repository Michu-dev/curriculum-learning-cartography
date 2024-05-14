import mlflow
import pandas as pd
import warnings


def read_experiments(experiment_names: list) -> pd.DataFrame:
    all_runs = mlflow.search_runs(experiment_names=experiment_names)
    cols = [
        "params.dataset",
        "params.ranking_mode",
        "params.ranked",
        "params.relaxed",
        "params.gamma",
        "metrics.test_acc",
        "metrics.f1_score",
        "metrics.roc_auc",
        "metrics.train_loss",
    ]
    experiments_df = all_runs[cols]
    cols_dict = {c: c[c.find(".") + 1 :] for c in cols}
    experiments_df.rename(columns=cols_dict, inplace=True)
    experiments_df["gamma"] = experiments_df["gamma"].astype(float)
    bool_conv_dict = {"True": True, "False": False}
    experiments_df["ranked"].replace(bool_conv_dict, inplace=True)
    experiments_df["relaxed"].replace(bool_conv_dict, inplace=True)
    return experiments_df


def get_methods_df(experiments_df: pd.DataFrame) -> pd.DataFrame:
    experiments_df = experiments_df[
        ~((experiments_df["relaxed"] == True) & (experiments_df["gamma"] >= 2.0))
    ]
    grouping_cols = ["dataset", "ranking_mode", "ranked", "relaxed"]
    agg_functions = {
        "test_acc": ["mean", "std"],
        "f1_score": ["mean", "std"],
        "train_loss": ["mean", "std"],
    }
    results_df = experiments_df.groupby(grouping_cols).agg(agg_functions)
    return results_df


def get_gammas_df(experiments_df: pd.DataFrame) -> pd.DataFrame:
    experiments_df = experiments_df[experiments_df["relaxed"] == True]
    print(experiments_df[experiments_df["dataset"] == "stellar"].shape)
    grouping_cols = ["dataset", "ranking_mode", "ranked", "gamma"]

    agg_functions = {
        "test_acc": ["mean", "std"],
        "f1_score": ["mean", "std"],
        "train_loss": ["mean", "std"],
    }
    results_df = experiments_df.groupby(grouping_cols).agg(agg_functions)
    return results_df


if __name__ == "__main__":
    warnings.filterwarnings("ignore")
    experiment_names = [
        "airline_ds_experiments",
        "credit_card_ds_experiments",
        "stellar_ds_experiments",
        "fashion_mnist_ds_experiments",
    ]
    experiments_df = read_experiments(experiment_names)
    methods_results_df = get_methods_df(experiments_df)

    print(methods_results_df)

    gammas_results_df = get_gammas_df(experiments_df)
    print(gammas_results_df)

    # experiment_names_dict = {dict(mlflow.get_experiment_by_name(e))['experiment_id']: e for e in experiment_names}
    # all_runs["experiment_name"] = all_runs.apply(lambda r: experiment_names_dict[r.experiment_id], axis=1)
    with pd.ExcelWriter("results.xlsx") as writer:
        methods_results_df.to_excel(writer, sheet_name="methods", index=True)
        gammas_results_df.to_excel(writer, sheet_name="gammas", index=True)
