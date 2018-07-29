import numpy as np
import os
import cv2
import glob

def load_data(folder):
    X_train = []
    Y_train = []
    X_test = []
    Y_test = []
    sample = list(np.arange(10))
    for i in range(40):
        folder_next = os.path.join(folder, 's%d' % (i+1))
        data = [cv2.imread(d, 0) for d in glob.glob(os.path.join(folder_next, '*.pgm'))]
        np.random.shuffle(sample)
        sample = sample[:5]
        X_train.extend([data[j].ravel() for j in range(10) if j in sample])
        Y_train.extend([i] * 5)
        X_test.extend([data[j].ravel() for j in range(10) if j not in sample])
        Y_test.extend([i] * 5)
    return np.array(X_train), np.array(Y_train), np.array(X_test), np.array(Y_test)
