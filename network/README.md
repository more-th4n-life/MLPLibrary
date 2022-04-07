## Network

The `network` directory contains 5 important modules: `activ.py`, `layer.py`, `loss.py`, `net.py` and `optim.py`, which hold classes directly used in the building, training and testing of models. The `network` directory also contains 3 key sub-directories: `dataset`, `loader` and `model`.

### Modules:
1. `activ.py`: keeps parent class `Activation`, and child classes `ReLU` and `LeakyReLU` (non-linear activation functions)
2. `layer.py`: keeps parent class `Layer`, and child class `Linear` (the hidden linear layers for a multi layer perceptron)
3. `loss.py`: keeps parent class `Loss`, and child class `CrossEntropyLoss` which is the criterion used for loss in training
4. `net.py`: keeps class **`Net`** for the construction, training, saving, loading and evaluation of MLP models built with MLPLibrary. 
5. `optim.py`: keeps the parent class `Optimizer`, and child classes `SGD` and `Adam` which are the optimizers currently supported.

### Sub-directories:
1. `dataset`: contains all datasets used for training/testing models, and also stores `source.py` with functions to load data as `np.array`.
2. `loader`: contains code for "processing" of data (e.g. normalization or PCA), and `data_loader` which houses all functions for loading the assignment data and allows for parametric selections that apply the use of data processing methods for ease of convenience.
3. `model`: contains all the models that are saved either explicitly when using the `Net` class method `save_model`, or automatically within the training process where `train_network` or `train_convergence` stores the *best model* found (one with lowest validation loss) for reloading and use in prediction.
