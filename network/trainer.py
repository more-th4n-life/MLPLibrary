from layer import *
from loss import *
from net import *
from optim import *
from loader.data_loader import *
from loader.test_loader import *

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

def network3():
    """
    Testing LeakyReLU and Random Samplers for Data Loaders
    """
    train_loader, valid_loader, test_loader = random_sample_loaders()

    mlp = Net(optimizer = SGD(0.1, 0.01), criterion=CrossEntropyLoss())

    mlp.add(Linear(128, 64))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 16))
    mlp.add(LeakyReLU())
    mlp.add(Linear(16, 10))

    mlp.train(train_loader, valid_loader, 11)
    mlp.test(test_loader)

def network2():
    """
    Testing deep network
    """
    train_loader, valid_loader, test_loader = example_loaders()

    mlp = Net(optimizer = SGD(0.1, 0.001), criterion=CrossEntropyLoss())

    mlp.add(Linear(128, 120))
    mlp.add(ReLU())
    mlp.add(Linear(120, 110))
    mlp.add(ReLU())
    mlp.add(Linear(110, 80))
    mlp.add(ReLU())
    mlp.add(Linear(80, 70))
    mlp.add(ReLU())
    mlp.add(Linear(70, 40))
    mlp.add(ReLU())
    mlp.add(Linear(40, 20))
    mlp.add(ReLU())
    mlp.add(Linear(20, 15))
    mlp.add(ReLU())
    mlp.add(Linear(15, 10))

    mlp.train(train_loader, valid_loader, 30)
    mlp.test(test_loader)

def network1():
    """
    Testing normal network
    """
    train_loader, valid_loader, test_loader = example_loaders()

    mlp = Net(optimizer = SGD(0.1, 0.001), criterion=CrossEntropyLoss())

    mlp.add(Linear(128, 64))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32))
    mlp.add(ReLU())
    mlp.add(Linear(32, 16))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))

    mlp.train(train_loader, valid_loader, 11)
    mlp.test(test_loader)

def main():
    network1()

if __name__ == "__main__":
    main()