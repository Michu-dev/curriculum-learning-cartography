# Curriculum Learning with automatic relaxation of loss function - Master's Thesis

## Project Description

The main task for the project was to find alternative version of curriculum learning which is the method where the data is provided on the input of a model in the order from easiest examples to the hardest. The automatic relaxation of loss function relies on the assumption that loss made by model on the hard examples isn't propagated in the early stage of learning process. It is taken into account in a weak way (relaxed means that the loss is a small part of its normal loss). Then, with the progress of learning, the loss is becoming more and more equal to the normal loss which comes from loss function e.g. Cross-Entropy.

Based on these 2 different approaches I've tried to test these methods, using Dataset Cartography and Self-Confidence from CleanLab to get the difficulty metric of each example from 4 different datasets:

- _Airline Passenger Satisfaction_ - binary classification, tabular data, balanced data
- _Credit Card Fraud Detection_ - binary classification, tabular data, highly imbalanced data
- _Stellar Classification Dataset_ - multiclass classification (3 target variable classes), tabular data, imbalanced
- _Fashion MNIST_ - multiclass classification, image data, balanced data

For the tabular and image data, 2 separate model architectures was created, using _Neural Network Intelligence_ tool for AutoML:

- Feed-Forward Architecture with embeddings for categorical variables for _tabular_ data
- Convolutional Neural Network Architecture for _image_ data (_Fashion MNIST_)

## Project Organization

    ├── LICENSE
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources. (Empty)
    │   ├── interim        <- Intermediate data that has been transformed. (Empty)
    │   ├── processed      <- The final, canonical data sets for modeling. (Empty)
    │   └── raw            <- The original, immutable data dump, tracked by DVC.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks for EDA.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis. Mainly relaxed loss function analysis.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download and preprocess data
    │   │   └── .env
    │   │   └── airline_passenger_satisfaction_train.py
    │   │   └── credit_card_fraud.py
    │   │   └── fashion_mnist.py
    │   │   └── spotify_tracks_genre.py
    │   │   └── stellar_ds.py
    │   │
    │   ├── features       <- Scripts necessary to run curriculum learning algorithms
    │   │   └── cartography_functions.py
    │   │   └── loss_function_relaxation.py
    │   │
    │   ├── models         <- Scripts to define benchmark neural network architectures
    |   |   ├── neural_architecture_search <- Scripts to run NAS and HO
    |   |   |   └── nas_cnn_model.py
    |   |   |   └── nas_ff_model.py
    |   |   |   └── README.md
    |   |   |   └── run_experiment.py
    │   │   ├── cnn_classifier.py
    │   │   └── generalised_neural_network_model.py
    │   │
    │   └── visualization  <- Scripts to create results oriented visualizations and conduct t-test
    │   |    └── experiments_agg.py
    │   ├── dataset_training.py <- script to setup data and models for training
    |   ├── train_model.py      <- script to orchestrate running experiments
    |   ├── train_run           <- script with functions for running training/validation loop
    ├── run_experiment.sh  <- script file to run series of experiments
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io

---

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>

## Running an experiment

To run the experiment, at first you have to run an instance of mlflow, exposed to a particular port and host address. Then you have to run _train_model_ script with the following options:

```
mlflow ui -p 5000 -h 0.0.0.0
python -m src.train_model -d stellar
python -m src.train_model -h
  -h, --help            show this help message and exit
  -d credit_card, --dataset credit_card
                        Dataset to use for NN model evaluation
  -r, --relaxed         Relax loss function flag
  -rnk, --ranked        Rank examples from training set in level of difficulty flag
  -b 1000, --batch-size 1000
                        Batch size of the data
  -e 8, --epochs 8      Epochs number of training
  -l 0.01, --lr 0.01    Learning rate of optimizer
  -p, --plot-map        Flag whether to plot and save cartography map for dataset
  -m None, --rank-mode None
                        Mode in which the experiment is running: Cartography, Self-confidence, None
  -a None, --alpha None
                        Alpha parameter of 2D -> 1D mapping in Cartography
  -bt None, --beta None
                        Beta parameter of 2D -> 1D mapping in Cartography
  -g 2.0, --gamma 2.0   Gamma parameter of relaxation loss function
```

The experiments was running and can be running using two difficulty of examples ranking: _cartography_ and _confidence_. The first metric has been built using mapping from 2D to 1D using _confidence_ and _variability_ metrics from _Dataset Cartography_ publication.

## Findings and key notes

The experiments with alone relaxation of loss function gained best results. Unfortunately, thanks to the AutoML results, only _Adam_ optimizer was used for all methods so ranking method (_curriculum learning_ with and without relaxed loss function) performed poorly. It is recommended to test ranking method with other optimizers or try to find solution for better performance in combination with _Adam_.
