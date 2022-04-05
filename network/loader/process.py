import numpy as np
from numpy import min, max, mean, std, load, array, eye


def train_test_split(data, label, ratio=0.2, shuffle=True):

    if shuffle:
        rand = np.random.permutation(data.shape[0])

    split = int(np.floor(ratio * data.shape[0]))

    # a ratio=0.2 means 20% test 80% training data

    train_data = data[rand[split:]] if shuffle else data[split:]
    train_label = label[rand[split:]] if shuffle else label[split:]

    test_data = data[rand[:split]] if shuffle else data[:split]
    test_label = label[rand[:split]] if shuffle else label[:split]
    
    return (train_data, train_label), (test_data, test_label)


def normalize(data):
    return (data - min(data)) / (max(data) - min(data))


def standardize(data):
    return (data - mean(data, axis=0)) / std(data, axis=0)


def one_hot(data, classes):
    label = array(data).reshape(-1)
    return eye(classes)[label]


def pca(data, n_comp=2):
    
    data_mean = (data - np.mean(data, axis=0))          # mean center
    cov_mat = np.cov(data_mean, rowvar=False)           # calculating covariance matrix of mean centered data
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)  # calculate eigenvalues and eigenvectors of the covariance mtrx
    sorted_index = np.argsort(eigen_values)[::-1]           # sort eigenvals in descending order
    sorted_eigenvectors = eigen_vectors[:,sorted_index]     # similarly sort eigenvectors
    eigenvector_subset = sorted_eigenvectors[:,0:n_comp]
    data_reduced = (eigenvector_subset.T @ data_mean.T).T

    return data_reduced     # return reduced n principal components

def identity(data):
    return data


def get_data():
    ld = lambda fn: load(fn)
    root = 'network/data/' 
    return \
    ld(root + "train_data.npy"), \
    ld(root + "train_label.npy"), \
    ld(root + "test_data.npy"), \
    ld(root + "test_label.npy")
