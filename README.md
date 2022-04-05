# MLPLibrary
A library for constructing MLP models, implemented with Numpy and Python.

Training code is in network/trainer.py. To run:
```
python3 network/trainer.py
```


### Installation:
Change into 'network/' directory and type:
```
pip install ..
```

To import library code:

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