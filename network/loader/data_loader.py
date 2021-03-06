from network.loader.process import standardize, normalize, get_data, one_hot, pca, train_test_split, identity
from network.dataset.source import get_data_from_file, get_data_from_url
import numpy as np


def load_train_val_test(method="standardize", n_categories=10, shuffle=True, ratio=0.2, pca_N=0, source="url"):
    
    get_data = {
        "url": get_data_from_url,
        "file": get_data_from_file
    }
    norm = {
        "standardize": standardize,
        "normalize": normalize,
        "none": identity
    }
    train_dat, train_lab, test_dat, test_lab = get_data[source]()  # reads in from numpy files in 'data/' directory

    if pca_N:
        train_dat = pca(train_dat, n_comp=pca_N)
        test_dat = pca(test_dat, n_comp=pca_N)

    else:
        train_dat = norm[method](train_dat)
        test_dat = norm[method](test_dat)

    if n_categories:
        train_lab = one_hot(train_lab, classes=n_categories)
        test_lab = one_hot(test_lab, classes=n_categories)

    train_set, valid_set = train_test_split(train_dat, train_lab, ratio=ratio, shuffle=shuffle)

    return train_set, valid_set, (test_dat, test_lab)
    

if __name__ == "__main__":
    data = get_data()
    train_data = data[0]
    train_label = one_hot(data[1], 10)
    test_data = data[2]
    test_label = one_hot(data[3], 10)