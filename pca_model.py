import numpy as np
from data_process import * 


def train_test_split(X, Y, test_size=0.2, shuffle=True, seed=None):
    n_samples = np.shape(X)[0]
    if shuffle:
        X_shuffle, Y_shuffle = shuffle_data(X, Y)
    n_train_samples = int(n_samples * (1 - test_size))
    x_train, y_train = X_shuffle[:n_train_samples], Y_shuffle[:n_train_samples]
    x_test, y_test = X_shuffle[n_train_samples:], Y_shuffle[n_train_samples:]
    return x_train, y_train, x_test, y_test

class PCA:
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
    
    def transform(self, X, K):
        X_mean = np.mean(X, 0)
        Z = X - np.tile(X_mean, (np.shape(X)[0], 1))
        covariance = np.cov(Z, rowvar=0)

        self.eigen_values, self.eigen_vectors = np.linalg.eig(np.mat(covariance))
        index = np.argsort(self.eigen_values)
        eigVals = self.eigen_values[index][:K]
        eigVects = self.eigen_vectors[index][:,:K]
        X_transformed = np.dot(Z, eigVects)
        return X_transformed, X_mean, eigVects
