from layer import *
from loss import *
from net import Net
from net import *
from optim import *
from activ import *
from loader.data_loader import *

import matplotlib.pyplot as pl

def network_mb():
    """
    Testing train_mb which doesn't use data and supports mini batch
    """
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = SGD(0.01, 0.001), criterion=CrossEntropyLoss())

    mlp.add(Linear(128, 64))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 16))
    mlp.add(LeakyReLU())
    mlp.add(Linear(16, 10))

    mlp.train_mb(train_set, val_set, 100, batch_size=20)


def network_bn2():
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = SGD(learning_rate=0.1, weight_decay=0, momentum=0.5), criterion=CrossEntropyLoss(), batch_norm=True)

    mlp.add(Linear(128, 1024))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32))
    mlp.add(ReLU())
    mlp.add(Linear(32, 16))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))

    CE = mlp.train_network(train_set=train_set, valid_set=val_set, epochs=100, batch_size=500)

    print(mlp.validate_batch(test_set[0], test_set[1], batch_size=500))

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

def network_batch_norm():
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = SGD(learning_rate=0.1, weight_decay=0, momentum=0.5), criterion=CrossEntropyLoss(), batch_norm=True)

    mlp.add(Linear(128, 1024))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 512))
    mlp.add(ReLU())
    mlp.add(Linear(512, 256))
    mlp.add(ReLU())
    mlp.add(Linear(256, 128))
    mlp.add(ReLU())
    mlp.add(Linear(128, 64))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32))
    mlp.add(ReLU())
    mlp.add(Linear(32, 16))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))

    CE = mlp.train_network(train_set=train_set, valid_set=val_set, epochs=100, batch_size=500)

    print(mlp.validate_batch(test_set[0], test_set[1], batch_size=1))

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

def network_adam_pca():
    n_comp = 96

    np.random.seed(10)
    train_set, val_set, test_set = pca_train_val_test(n_comp = n_comp)  # take first 16 components

    mlp = Net(optimizer = Adam(), criterion=CrossEntropyLoss(), L2_reg_term=0, batch_norm=False)

    mlp.add(Linear(n_comp, 1024))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64))
    mlp.add(ReLU())
    mlp.add(Linear(64, 10))

    CE = mlp.train_network(train_set=train_set, valid_set=val_set, epochs=200, batch_size=500)

    print(mlp.validate_batch(test_set[0], test_set[1], batch_size=500))

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

def network_convergence():
    np.random.seed(10)
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = Adam(learning_rate=0.001), criterion=CrossEntropyLoss(), L2_reg_term=0, batch_norm=False)

    mlp.add(Linear(128, 1024, weights="xavier", bias="const", dropout=0.3))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="xavier", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="xavier", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="xavier", bias="const"))

    mlp.set_name("test_model")

    CE, _, _, _ = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=1000, report_interval=1, last_check=10, threshold=10)

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

    print(mlp.ep)

def network_adam():
    np.random.seed(10)
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = Adam(learning_rate=0.001), criterion=CrossEntropyLoss(), L2_reg_term=0, batch_norm=False)

    mlp.add(Linear(128, 1024, weights="xavier", bias="const", dropout=0.3))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="xavier", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="xavier", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="xavier", bias="const"))

    CE = mlp.train_network(train_set=train_set, valid_set=val_set, epochs=60, batch_size=500)

    print(mlp.validate_batch(test_set[0], test_set[1], batch_size=500))

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

def network1():
    """
    Testing normal network
    """
    np.random.seed(10)
    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = SGD(learning_rate=0.04, momentum=0.5), criterion=CrossEntropyLoss(), L2_reg_term=0.004, batch_norm=True)

    mlp.add(Linear(128, 1024, dropout=0.3))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(64, 16, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))

    CE = mlp.train_network(train_set=train_set, valid_set=val_set, epochs=500, batch_size=500)

    print(mlp.validate_batch(test_set[0], test_set[1], batch_size=500))

    pl.figure(figsize=(15,4))
    pl.plot(CE)
    pl.grid()
    pl.show()

def load_model():
    model = Net.load_model("network/model/test_model")
    print(model)

def main():
    network_convergence()

if __name__ == "__main__":
    main()