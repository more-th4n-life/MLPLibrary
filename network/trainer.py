from layer import *
from loss import *
from net import Net
from net import *
from optim import *
from activ import *
from loader.data_loader import *

import matplotlib.pyplot as pl


def plot_results(ep, tl, ta, vl, va):

    fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(2, 2)
    fig.suptitle(f'Training Results, best model found @ Epoch {ep}')

    ax1.plot(tl)
    ax1.set_title('Training Loss')

    ax2.plot(vl, 'tab:orange')
    ax2.set_title('Validation Loss')

    ax3.plot(ta, 'tab:green')
    ax3.set_title('Training Accuracy')

    ax4.plot(tl, 'tab:red')
    ax4.set_title('Validation Accuracy')
    
    for ax in fig.get_axes():
        ax.label_outer()

    pl.show()

def network_train():
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

    mlp.set_name("test_model_train")

    ep, tl, ta, vl, va = mlp.train_network(train_set=train_set, valid_set=val_set, batch_size=500, epochs=10)
    plot_results(ep, tl, ta, vl, va)

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
                                                report_interval=5, last_check=10, threshold=0.01)

    plot_results(ep, tl, ta, vl, va)

def class_output (y_hat):
    y_pred = np.argmax(y_hat, axis=1)
    return y_pred

def load_model_test():
    model = Net.load_model("network/model/test_model_train")
    train_set, val_set, test_set = train_val_test()

    model.test_network(test_set)

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
    #network_convergence()
    #network_train()
    #load_model_predict()
    load_model_test()


if __name__ == "__main__":
    main()