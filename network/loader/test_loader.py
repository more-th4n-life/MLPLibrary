from ast import Sub
from webbrowser import get
from loader.data_loader import DataLoader, Loader, collate, worker_func, Dataset
from loader.process import standardize, normalize, get_data, one_hot, pca
import numpy as np
from loader.sampler import SubsetRandSampler

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

def random_sample_loaders():
    data = get_data()
    train_data = standardize(data[0])
    train_label = one_hot(data[1], 10)
    test_data = standardize(data[2])
    test_label = one_hot(data[3], 10)

    data = Dataset(train_data, train_label)
    test = Dataset(test_data, test_label)

    num_train = len(data)
    idx = list(range(num_train))
    val_size = 0.2
    spl = int(np.floor(val_size * num_train))

    train_idx = idx[spl:]
    valid_idx = idx[:spl]
    train_sampler = SubsetRandSampler(train_idx)
    valid_sampler = SubsetRandSampler(valid_idx)

    train_loader = DataLoader(data[spl:], n_workers=1, batch_size=64)
    valid_loader = DataLoader(data[:spl], n_workers=1, batch_size=64)
    test_loader = DataLoader(test, n_workers=1, batch_size=64)

    return train_loader, valid_loader, test_loader


def example_loaders():
    data = get_data()
    train_data = standardize(data[0])
    train_label = one_hot(data[1], 10)
    test_data = standardize(data[2])
    test_label = one_hot(data[3], 10)

    data = Dataset(train_data, train_label)
    test = Dataset(test_data, test_label)

    num_train = len(data)
    #idx = list(range(num_train))
    val_size = 0.2
    spl = int(np.floor(val_size * num_train))

    #train_idx = idx[spl:]
    #valid_idx = idx[:spl]
    #train_sampler = SubsetRandSampler(train_idx)
    #valid_sampler = SubsetRandSampler(valid_idx)

    train_loader = DataLoader(data[spl:], n_workers=1, batch_size=64)
    valid_loader = DataLoader(data[:spl], n_workers=1, batch_size=64)
    test_loader = DataLoader(test, n_workers=1, batch_size=64)

    return train_loader, valid_loader, test_loader


if __name__ == "__main__":
    data = get_data()
    train_data = data[0]
    train_label = one_hot(data[1], 10)
    test_data = data[2]
    test_label = one_hot(data[3], 10)

    data = Dataset(train_data, train_label)
    test = Dataset(test_data, test_label)

    num_train = len(data)
    #idx = list(range(num_train))
    val_size = 0.2
    spl = int(np.floor(val_size * num_train))
    print(spl)
    #train_idx = idx[spl:]
    #valid_idx = idx[:spl]
    #train_sampler = SubsetRandSampler(train_idx)
    #valid_sampler = SubsetRandSampler(valid_idx)

    train_loader = DataLoader(data[spl:], n_workers=2, batch_size=64)
    valid_loader = DataLoader(data[:spl], n_workers=2, batch_size=64)
    test_loader = DataLoader(test, n_workers=2, batch_size=64)

    x, y = next(train_loader)
    print(x.shape)
    print(y.shape)
    print(len(y))

    #train_loader = DataLoader(data[spl:], n_workers=4)
    #valid_loader = DataLoader(data[:spl])
    #test_loader = DataLoader(test)

    """
    idx = list(range(num_train))
    np.random.shuffle(idx)

    valid_size = 0.2

    split = int(np.floor(valid_size * num_train))
    train_idx, valid_idx = idx[split:], idx[:split]

    train_loader = DataLoader(train[train_idx])
    valid_loader = DataLoader(train[valid_idx])
    test_loader = DataLoader(test)
    """