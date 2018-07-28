import numpy as np

def load_data(filename):
    pass

def shuffle_data(X, Y, seed=None):
    if seed:
        np.random.seed(seed)
    index = np.arange(X.shape[0])
    return X[index], Y[index]
