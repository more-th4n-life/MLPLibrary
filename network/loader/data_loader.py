from loader.process import standardize, normalize, get_data, one_hot, pca, train_test_split, identity
import numpy as np


def load_train_val_test(method="standardize", n_categories=10, shuffle=True, pca_N=0):
    
    norm = {
        "standardize": standardize,
        "normalize": normalize,
        "none": identity
    }
    train_data, train_label, test_data, test_label = get_data()  # reads in from numpy files in 'data/' directory

    if pca_N:
        train_data = pca(train_data, n_comp=pca_N)
        test_data = pca(test_data, n_comp=pca_N)

    else:
        train_data = norm[method](train_data)
        test_data = norm[method](test_data)

    if n_categories:
        train_label = one_hot(train_label, classes=n_categories)
        test_label = one_hot(test_label, classes=n_categories)

    train_set, valid_set = train_test_split(train_data, train_label, ratio=0.2, shuffle=shuffle)

    return train_set, valid_set, (test_data, test_label)
    

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