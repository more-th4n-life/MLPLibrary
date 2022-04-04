from re import L
from layer import *
from loss import *
from net import Net
from net import *
from optim import *
from activ import *
from loader.data_loader import *

import matplotlib.pyplot as pl

np.random.seed(42)


def plot_results(ep, tl, ta, vl, va):

    fig, ((ax1, ax2), (ax3, ax4)) = pl.subplots(2, 2)
    fig.suptitle(f'Training Results, best model found @ Epoch {ep}')

    ax1.plot(tl)
    ax1.set_title('Training Loss')

    ax2.plot(vl, 'tab:orange')
    ax2.set_title('Validation Loss')

    ax3.plot(ta, 'tab:green')
    ax3.set_title('Training Accuracy')

    ax4.plot(va, 'tab:red')
    ax4.set_title('Validation Accuracy')
    
    for ax in fig.get_axes():
        ax.label_outer()

    pl.show()


def load_model_test(name="deep_skinny"):
    model = Net.load_model("network/model/" + name)
    train_set, val_set, test_set = load_train_val_test()

    model.test_network(train_set, "train data")
    model.test_network(val_set, "valid data")
    model.test_network(test_set, "test data")


def network_train():
    train_set, val_set, _ = load_train_val_test()

    mlp = Net(optimizer = SGD(learning_rate=0.1, momentum=0.9, lr_decay="exp"), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=False)

    # training time: (chosen 100 epochs) ... validation loss increases rapidly after ep 50 (training loss miniscule)
    #  
    # accuracy on train:  89.23%
    # accuracy on valid: 89.29%
    # accuracy on test: 46.15%
    # 
    # best model @ epoch 6!!!!

    mlp.add(Linear(128, 1024, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="xavier", bias="const"))

    mlp.set_name("test_sgd_exp_train")

    ep, tl, ta, vl, va = mlp.train_network(train_set=train_set, valid_set=val_set, batch_size=500, epochs=100, report_interval=1)
    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def network_train_adam():
    train_set, val_set, _ = load_train_val_test()

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=False)

    # training time: 32.6 secs (chosen 10 epochs)
    #  
    # accuracy on train: 80.42%
    # accuracy on valid: 80.36%
    # accuracy on test: 52.17%
    # 
    # best model @ epoch 3

    mlp.add(Linear(128, 1024, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="xavier", bias="const"))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="xavier", bias="const"))

    mlp.set_name("test_adam_train")

    ep, tl, ta, vl, va = mlp.train_network(train_set=train_set, valid_set=val_set, batch_size=500, epochs=10)
    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def network_convergence():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = SGD(learning_rate=0.1, momentum=0.9, lr_decay="default"), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.004, batch_norm=True)

    # training time: 1min 54.4secs
    #  
    # accuracy on train: 64.03%
    # accuracy on valid: 64.46%
    # accuracy on test: 46.27%
    # 
    # best model @ epoch 17

    mlp.add(Linear(128, 1024, weights="kaiming", bias="const", dropout=0))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64, weights="kaiming", bias="const", dropout=0))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32, weights="kaiming", bias="const", dropout=0))
    mlp.add(ReLU())
    mlp.add(Linear(32, 10, weights="kaiming", bias="const", dropout=0))
    

    mlp.set_name("converged_model")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def network_convergence2():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = SGD(learning_rate=0.1, weight_decay=0.01, momentum=0.9, lr_decay="exp"), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.004, batch_norm=True)

    # training time: 2min 55sec (GOOD MODEL!)
    #  
    # accuracy on train: 70.51%
    # accuracy on valid: 71.43%
    # accuracy on test: 55.35%
    # 
    # best model @ epoch 33

    mlp.add(Linear(128, 1024, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(32, 10, weights="kaiming", bias="const", dropout=0))
    

    mlp.set_name("converged_model2")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.000001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def network_convergence3():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = SGD(learning_rate=0.05, weight_decay=0.01, momentum=0.5, lr_decay="default"), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 5min 58.6sec (GOOD MODEL TO SHOW)
    #  
    # accuracy on train: 81.38%
    # accuracy on valid: 81.70%
    # accuracy on test: 53.67%
    # 
    # best model @ epoch 21

    mlp.add(Linear(128, 2048, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(2048, 128, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(128, 16, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))
    

    mlp.set_name("converged_model3")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)



def network_convergence_adam():

    train_set, val_set, test_set = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(), criterion=CrossEntropyLoss(), L2_reg_term=0, batch_norm=False)

    # training time: 53.3s
    #  
    # accuracy on train: 65.62%
    # accuracy on valid: 66.22%
    # accuracy on test: 50.46%
    # 
    # best model @ epoch 16

    mlp.add(Linear(128, 1024, weights="kaiming", bias="const", dropout=0))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64, weights="kaiming", bias="const", dropout=0))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 10, weights="kaiming", bias="const", dropout=0))

    mlp.set_name("adam_model")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=10, last_check=10, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)
    

def network_convergence_adam2():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 3min 48.6sec (GOOD ADAM MODEL)
    #  
    # accuracy on train: 78.15%
    # accuracy on valid: 77.78%
    # accuracy on test: 54.72%
    # 
    # best model @ epoch 8

    mlp.add(Linear(128, 2048, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(2048, 128, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(128, 16, dropout=0.1))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))
    
    mlp.set_name("adam_model2")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def network_convergence_adam3():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 9min 11.3sec (GOOD ADAM MODEL - WIDE)
    #  
    # accuracy on train: 77.58%
    # accuracy on valid: 76.91%
    # accuracy on test: 55.87%
    # 
    # best model @ epoch 7

    mlp.add(Linear(128, 4096, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(4096, 256, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(256, 32, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(32, 10))
    
    mlp.set_name("adam_model3")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def network_convergence_adam4():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.0001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 24min 56.2sec (WIDE & DEEP ADAM MODEL) VERY GOOD - JUST LONG TO TRAIN
    #  
    # accuracy on train: 82.69%
    # accuracy on valid: 83.33%
    # accuracy on test: 55.77%
    # 
    # best model @ epoch 49

    mlp.add(Linear(128, 4096, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(4096, 256, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(256, 32, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(32, 16, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))
    
    mlp.set_name("adam_model4")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

"""""""""""""""""""""""""""""""""""""""""""""
    MODEL Accu: 55.95% - TIME 8 min 11 sec
"""""""""""""""""""""""""""""""""""""""""""""

def network_convergence_adam5():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 8min 11.6sec   NOT AS LONG TO TRAIN DUE TO HIGHER LR
    #  
    # accuracy on train: 77.18%
    # accuracy on valid: 77.02%
    # accuracy on test: 55.95%
    # 
    # best model @ epoch 9

    mlp.add(Linear(128, 4096, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(4096, 512, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(512, 256, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(256, 64, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(64, 16, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))
    
    mlp.set_name("adam_model5")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""
DEEPEST MODEL Accu: 54.10% - TIME 40 min 36.6 sec 
"""""""""""""""""""""""""""""""""""""""""""""""""""""""""""

def network_convergence_adam6():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 40 min 36.6 sec (DEEPEST ADAM MODEL) VERY LONG TO TRAIN/PREDICT
    #  
    # accuracy on train: 72.53%     
    # accuracy on valid: 72.25%
    # accuracy on test: 54.10%
    # 
    # best model @ epoch 5

    mlp.add(Linear(128, 8192, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(8192, 4096, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(4096, 512, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(512, 256, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(256, 64, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(64, 16, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(16, 10))
    
    mlp.set_name("adam_model6")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.00001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def shallow_wide_convergence():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = SGD(learning_rate=0.05, weight_decay=0.01, momentum=0.9, lr_decay="exp"), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 30 min, 37.7 s           WIDE IS LONG TO TRAIN
    #  
    # accuracy on train: 76.30%
    # accuracy on valid: 54.34%
    # accuracy on test: 54.28%
    # 
    # best model @ epoch 14

    mlp.add(Linear(128, 16384, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(16384, 128, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(128, 10, weights="kaiming", bias="const"))
    

    mlp.set_name("shallow_wide")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def deep_skinny_convergence():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 5 min 19.5 sec
    #  
    # accuracy on train: 72.79%
    # accuracy on valid: 54.60%
    # accuracy on test: 54.81%
    # 
    # best model @ epoch 16

    mlp.add(Linear(128, 1024, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 512, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(512, 512, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(512, 256, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(256, 256, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(256, 128, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(128, 64, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 10, weights="kaiming", bias="const"))

    mlp.set_name("deep_skinny")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def deep_adam_convergence():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=True)

    # training time: 11 min 3.2 sec
    #  
    # accuracy on train: 69.54 %
    # accuracy on valid: 69.53 %
    # accuracy on test: 53.60 %
    #
    # best model @ epoch 14

    mlp.add(Linear(128, 2048, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(2048, 1024, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 512, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(512, 256, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(256, 128, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(128, 64, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 16, weights="kaiming", bias="const", dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(16, 10, weights="kaiming", bias="const"))

    mlp.set_name("adam_deep")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def deep_adam_convergence2():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.004, batch_norm=True)

    # training time: 16 min 29.5 sec
    #  
    # accuracy on train: 69.98
    # accuracy on valid: 70.68
    # accuracy on test: 53.11
    #
    # best model @ epoch 17

    mlp.add(Linear(128, 2048, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(2048, 1024, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 516, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(516, 1024, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 256, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(256, 516, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(516, 128, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(128, 256, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(256, 64, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 32, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(32, 16, dropout=0.2))
    mlp.add(LeakyReLU())
    mlp.add(Linear(16, 10))

    mlp.set_name("adam_deep2")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=2, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def small_network_adam():

    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.001, batch_norm=False)

    # training time: 47.4 sec
    #  
    # accuracy on train: 67.41
    # accuracy on valid: 67.71
    # accuracy on test: 53.46
    #
    # best model @ epoch 3

    mlp.add(Linear(128, 1024))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 10))

    mlp.set_name("adam_small")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)

def small_network_adam2():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = Adam(learning_rate=0.00001), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.01, batch_norm=False)

    # training time: 17 min 26.4 sec           Very low LR set (small steps but adam adaptive)
    #  
    # accuracy on train: 70.63
    # accuracy on valid: 70.41
    # accuracy on test: 54.71
    #
    # best model @ epoch 3

    mlp.add(Linear(128, 1024))
    mlp.add(LeakyReLU())
    mlp.add(Linear(1024, 64))
    mlp.add(LeakyReLU())
    mlp.add(Linear(64, 10))

    mlp.set_name("adam_small2")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def small_network_sgd():
    train_set, val_set, _ = load_train_val_test(method="standardize", shuffle=True, n_categories=10)

    mlp = Net(optimizer = SGD(learning_rate=0.04, weight_decay=0.001, momentum=0.999), \
                criterion=CrossEntropyLoss(), L2_reg_term=0.01, batch_norm=True)

    # training time: 5 min, 22.7 s   + 30 MORE EPOCHS: 2 min, 33.6 s   +  30 EPOCHS +             10 EPOCHS 
    #  
    # accuracy on train: 70.41                              83.57%             86.31%               86.80      (overfitting on test)
    # accuracy on valid: 77.67                              83.72%             86.07%               87.61      (overfitting on valid)
    # accuracy on test: 54.55 (I am going to run longer)    55.01%             55.01%               54.77       ^ more interesting
    #
    # best model @ epoch 57                               epoch 11 / 30        epoch 0 / 30       epoch 5 / 30

    mlp.add(Linear(128, 1024, dropout=0.2))
    mlp.add(ReLU())
    mlp.add(Linear(1024, 64))
    mlp.add(ReLU())
    mlp.add(Linear(64, 32))
    mlp.add(ReLU())
    mlp.add(Linear(32, 10))

    mlp.set_name("sgd_small")

    ep, tl, ta, vl, va = mlp.train_convergence(train_set=train_set, valid_set=val_set, batch_size=500, planned_epochs=10000, \
                                                report_interval=1, last_check=10, threshold=0.0001)

    plot_results(ep, tl, ta, vl, va)

    load_model_test(mlp.model_name)


def continue_small_sgd():
    model = Net.load_model("network/model/" + "sgd_small")
    train_set, val_set, test_set = load_train_val_test()
    model.train_network(train_set, val_set, batch_size=500, epochs=10, report_interval=1)
    
    load_model_test(model.model_name)


def main():
    #network_convergence_adam()  
    #network_train_adam()
    #network_train()

    #network_convergence()
    #network_convergence2()
    #network_convergence3()

    #network_convergence_adam
    #network_convergence_adam2 
    #network_convergence_adam3()  
    #network_convergence_adam4()    
    #network_convergence_adam5()    # BEST
    #network_convergence_adam6()    # DEEPEST + WIDEST

    #shallow_wide_convergence()  # WIDEST
    #deep_skinny_convergence()   

    #deep_adam_convergence() 
    #deep_adam_convergence2()   # DEEPEST
    
    #small_network_adam()
    #small_network_sgd()
    continue_small_sgd()
    #load_model_predict()
    #load_model_test()


if __name__ == "__main__":
    main()