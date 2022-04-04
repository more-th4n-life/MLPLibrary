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

def plot_results(ep, tl, ta, vl, va):

    fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(2, 2)
    fig.suptitle(f'Training Results, best model found @ Epoch {ep}')
    ax1.plot(tl)
    ax1.set_title('Training Loss')
    #ax1.xlabel('Epochs run')
    #ax1.ylabel('Loss')
    ax2.plot(vl, 'tab:orange')
    ax2.set_title('Validation Loss')
    #ax2.xlabel('Epochs run')
    #ax2.ylabel('Loss')
    ax3.plot(ta, 'tab:green')
    ax3.set_title('Training Accuracy')
    #ax3.xlabel('Epochs run')
    #ax3.ylabel('Accuracy')
    ax4.plot(tl, 'tab:red')
    ax4.set_title('Validation Accuracy')
    #ax4.xlabel('Epochs run')
    #ax4.ylabel('Accuracy')

    for ax in fig.get_axes():
        ax.label_outer()
        #ax.axvline(x = ep)

    pl.show()

def network_convergence():

    train_set, val_set, test_set = train_val_test()

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=False)

    mlp.add(Linear(128, 1024, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="xavier", bias="const"))

    mlp.set_name("test_model")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=5, last_check=20, threshold=0.001)

    plot_results(ep, tl, ta, vl, va)

def network_adam():

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

def class_output (y_hat):
    y_pred = np.argmax(y_hat, axis=1)
    return y_pred

def load_model_predict():

    model = Net.load_model("network/model/test_model")
    train_set, val_set, test_set = train_val_test()

    pred = model.predict(train_set[0], 10)
    pred, target = np.argmax(pred, axis=1), np.argmax(train_set[1], axis=1)

    correct = np.sum(pred==target)
    print("Count: ", pred.shape[0])
    print("Matches: ", correct)

    print(f"Accuracy on trained data: ", correct/pred.shape[0])

    pred = model.predict(val_set[0], 10)
    pred, target = np.argmax(pred, axis=1), np.argmax(val_set[1], axis=1)

    correct = np.sum(pred==target)
    print("Count: ", pred.shape[0])
    print("Matches: ", correct)

    print(f"Accuracy on valid data: ", correct/pred.shape[0])

    pred = model.predict(test_set[0], 10)
    pred, target = np.argmax(pred, axis=1), np.argmax(test_set[1], axis=1)

    correct = np.sum(pred==target)
    print("Count: ", pred.shape[0])
    print("Matches: ", correct)

    print(f"Accuracy on test data: ", correct/pred.shape[0])

def main():
    network_convergence()
    load_model_predict()

if __name__ == "__main__":
    main()