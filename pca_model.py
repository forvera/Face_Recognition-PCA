import numpy as np
import data_process 


def normalize(X, axis=-1, p=2):

    pass

def standardize(X):
    pass

def train_test_split(X, Y, test_size=0.2, shuffle=True, seed=None):
    n_samples = np.shape(X)[0]
    if shuffle:
        X_shuffle, Y_shuffle = shuffle_data(X, Y)
    n_train_samples = int(n_samples * (1 - test_size))
    x_train, y_train = X_shuffle[:n_train_samples], Y_shuffle[:n_train_samples]
    x_test, y_test = X_shuffle[n_train_samples:], Y_shuffle[n_train_samples:]
    return x_train, y_train, x_test, y_test
    
def calculate_covariance_matrix(X, Y=np.empty((0, 0))):
    if not Y.any():
        Y = X
    return np.mat(np.cov(X, Y, rowvar=0))

def calculate_variance(X):
    variance = np.var(X, axis=0)
    return variance

def calculate_std_dev(X):
    return np.sqrt(calculate_variance(X))

class PCA()：
    def __init__(self):
        self.eigen_values = None
        self.eigen_vectors = None
        self.K = 2
    
    def transform(self, X):
        covariance = calculate_covariance_matrix(X):

        self.eigen_values, self.eigen_vectors = np.linalg.eig(covariance)
        index = np.argsort(self.eigen_values)
        eigVals = self.eigen_values[index][:self.K]
        eigVects = self.eigen_vectors[index][:,:self.K]

        X_transformed = np.dot(X, eigVects)
        return X_transformed
