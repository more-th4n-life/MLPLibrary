# MLPLibrary
A library for constructing MLP models, implemented with Numpy and Python. \
\
Please refer to our detailed Sphinx auto-generated documentation found within the 
```docs/build/html``` sub-directory. 

## Installing MLPLibrary
To install MLPLibrary for use locally anywhere on your filesystem, navigate to
the main root folder ```MLPLibrary/``` and execute the following command in terminal:
```
pip install .
``` 
This will install all required dependencies, and allow use of our library anywhere on your 
device. To uninstall simply type ```pip uninstall MLPLibrary```.

## Running our Code
For practical demos of our code, navigate to the ```examples/``` directory. Here you will
find several python notebook experiments that execute MLPLibrary code for building, training
and evaluation of MLP models. To access our best model for *COMP5329 Assignment 1*, please 
refer to the `Best-Model.ipynb` notebook. This contains the code for our final model that was 
decided upon through numerous experiments within adjacent notebooks of this directory. Our best
model is saved (pickled) in the ```examples/model``` sub-directory, and is appropriately named,
`Best_model`. For code with instructions that demo building, training and evaluation of models 
please open ```Demo-MLP.ipynb``` which steps through in more detail, how to use MLPLibrary functions
to construct 2 MLP models, i.e. a model using SGD optimizer, and another using the Adam optimizer.

## MLPLibrary modules
The main modules of MLPLibrary include the `Net` class which is the focus of MLPLibrary in representing
a Neural Network that supports the building of *deep models*. Currently our modularised library 
functions only support the building of basic Multi-Layer Perceptron Models. The `Linear` class 
currently is the only supported `Layer` class (however, we plan to write our own `CNN` class in later 
releases of MLPLibrary as a project). `ReLU` & `LeakyReLU` are the only supported `Activation` functions,
and `CrossEntropyLoss` (with Softmax activation applied on last layer) the only `Loss` Criterion. `Optimizer` 
algorithms supported for the updating of weights and biases in each layer include both `SGD` and `Adam`. 

## To run driver code
`trainer.py` contains driver code used whilst developing and testing our library functions.
Functions within this file contain some sample code for building, training and testing models. 
```
python3 trainer.py
```
For more detailed documentation regarding our library please refer to ```docs/build/html``` sub-directory.

## Importing library code
This assumes pip installation if not within the root folder:
```
from network.net import Net
from network.layer import Linear
from network.loss import CrossEntropyLoss
from network.activ import ReLU, LeakyReLU
from network.optim import SGD, Adam

from network.loader.process import train_test_split, normalize, standardize, one_hot, pca, identity
from network.dataset.source import get_data_from_file, get_data_from_url
from network.loader.data_loader import load_train_val_test
```
