from nni.experiment import Experiment
import sys

search_space = {
    "batch_size": {
        "_type": "choice",
        "_value": [16, 32, 64, 128, 256, 512, 1024, 2048, 5096, 10192],
    },
    "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
    "weight_decay": {"_type": "uniform", "_value": [0, 0.1]},
    "optimizer": {"_type": "choice", "_value": ["SGD", "Adam"]},
    "hidden_layers": {"_type": "choice", "_value": [1, 2]},
    "dropout": {"_type": "uniform", "_value": [0.1, 0.9]},
    "emb_dropout": {"_type": "uniform", "_value": [0.2, 0.6]},
    "features": {"_type": "randint", "_value": [50, 500]},
}

experiment = Experiment("local")
dataset = sys.argv[1]
if dataset == "fashion_mnist":
    search_space = {
        "batch_size": {
            "_type": "choice",
            "_value": [16, 32, 64, 128, 256, 512, 1024, 2048, 5096],
        },
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
        "hidden_units": {"_type": "randint", "_value": [10, 150]},
        "weight_decay": {"_type": "uniform", "_value": [0, 0.1]},
        "optimizer": {"_type": "choice", "_value": ["SGD", "Adam"]},
        "conv_layers": {"_type": "choice", "_value": [1, 2]},
        "dense_layers": {"_type": "choice", "_value": [1, 2]},
        "kernel_conv": {"_type": "choice", "_value": [3, 4, 5]},
        "kernel_pooling": {"_type": "choice", "_value": [2, 3, 4]},
        "stride_conv": {"_type": "choice", "_value": [1, 2]},
        "stride_pooling": {"_type": "choice", "_value": [1, 2]},
        "padding_conv": {"_type": "choice", "_value": [0, 1, 2]},
        "padding_pooling": {"_type": "choice", "_value": [0, 1]},
        "dropout": {"_type": "uniform", "_value": [0.1, 0.9]},
        "features": {"_type": "randint", "_value": [50, 200]},
    }
    experiment.config.trial_command = (
        "python -m src.models.neural_architecture_search.nas_cnn_model"
    )
else:
    search_space = {
        "batch_size": {
            "_type": "choice",
            "_value": [16, 32, 64, 128, 256, 512, 1024, 2048, 5096, 10192],
        },
        "lr": {"_type": "loguniform", "_value": [0.0001, 0.1]},
        "weight_decay": {"_type": "uniform", "_value": [0, 0.1]},
        "optimizer": {"_type": "choice", "_value": ["SGD", "Adam"]},
        "hidden_layers": {"_type": "choice", "_value": [1, 2]},
        "dropout": {"_type": "uniform", "_value": [0.1, 0.9]},
        "emb_dropout": {"_type": "uniform", "_value": [0.2, 0.6]},
        "features": {"_type": "randint", "_value": [50, 500]},
    }
    experiment.config.trial_command = (
        f"python -m src.models.neural_architecture_search.nas_ff_model {dataset}"
    )
experiment.config.trial_code_directory = "."
experiment.config.search_space = search_space
experiment.config.tuner.name = "TPE"
experiment.config.tuner.class_args["optimize_mode"] = "maximize"
experiment.config.max_trial_number = 25
experiment.config.trial_concurrency = 1
# experiment.config.max_experiment_duration = "20m"
experiment.run(8081)
results = sorted(experiment.export_data(), key=lambda t: t.value, reverse=True)
for e in results[:3]:
    print(e)
input("Press enter to quit")
experiment.stop()
