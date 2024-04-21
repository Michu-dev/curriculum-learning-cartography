### Neural Architecture Search using NNI

The folder contains scripts to run Neural Architecture Search for datasets specified for the project purposes. The search spaces include parameters for Feed-Forward and simple Convolutional Neural Network model skeletons. All the paramerers are specifide inside `run_experiment.py` file. Search spaces have also parameters defined for hyperparameter optimization, like optimizer or learning rate. To start model NAS, run the following command:

```
python run_experiment.py {dataset_name}
```

where _dataset_name_ should be the name defined for the dataset used in the _Curriculum Learning with automatic relaxation of the loss function_ project: _airline_passenger_satisfaction_, _credit_card_, _spotify_tracks_, _stellar_ or _fashion_mnist_.
