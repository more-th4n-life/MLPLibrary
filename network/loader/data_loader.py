from loader.process import standardize, normalize, get_data, one_hot, pca
import numpy as np


def train_val_test():
    data = get_data()
    train_data = standardize(data[0])
    train_label = one_hot(data[1], 10)
    test_data = standardize(data[2])
    test_label = one_hot(data[3], 10)

    val_size = 0.2
    spl = int(np.floor(val_size * train_data.shape[0]))
    train_data, valid_data = train_data[spl:], train_data[:spl]
    train_label, valid_label = train_label[spl:], train_label[:spl]

    train_set = (train_data, train_label)
    valid_set = (valid_data, valid_label)
    test_set = (test_data, test_label)

    return train_set, valid_set, test_set
    

def pca_train_val_test(n_comp):
    data = get_data()
    train_data = pca(data[0], n_comp)
    train_label = one_hot(data[1], 10)
    test_data = pca(data[2], n_comp)
    test_label = one_hot(data[3], 10)

    val_size = 0.2
    spl = int(np.floor(val_size * train_data.shape[0]))
    train_data, valid_data = train_data[spl:], train_data[:spl]
    train_label, valid_label = train_label[spl:], train_label[:spl]

    train_set = (train_data, train_label)
    valid_set = (valid_data, valid_label)
    test_set = (test_data, test_label)

    return train_set, valid_set, test_set


if __name__ == "__main__":
    data = get_data()
    train_data = data[0]
    train_label = one_hot(data[1], 10)
    test_data = data[2]
    test_label = one_hot(data[3], 10)