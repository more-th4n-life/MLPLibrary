import numpy as np
from numpy import min, max, mean, std, load, array, eye

def normalize(data):
    return (data - min(data)) / (max(data) - min(data))

def standardize(data):
    return (data - mean(data, axis=0)) / std(data, axis=0)

def one_hot(data, classes):
    label = array(data).reshape(-1)
    return eye(classes)[label]

def pca(data, n_comp=2):
    data_mean = (data - np.mean(data, axis=0)) # mean center
    #D_m = D_m / np.std(D_m, axis=0)
    cov_mat = np.cov(data_mean, rowvar=False)  # calculating covariance matrix of mean centered data
    eigen_values , eigen_vectors = np.linalg.eigh(cov_mat)  # calculate eigenvalues and eigenvectors of the covariance mtrx
    sorted_index = np.argsort(eigen_values)[::-1]  # sort eigenvals in descending order
    #sorted_eigenvalues = eigen_values[sorted_index]
    sorted_eigenvectors = eigen_vectors[:,sorted_index]  # similarly sort eigenvectors
    eigenvector_subset = sorted_eigenvectors[:,0:n_comp]
    data_reduced = (eigenvector_subset.T @ data_mean.T).T
    return data_reduced
    
def get_data():
    ld = lambda fn: load(fn)
    root = "./library/data/"
    return \
    ld(root + "train_data.npy"), \
    ld(root + "train_label.npy"), \
    ld(root + "test_data.npy"), \
    ld(root + "test_label.npy")