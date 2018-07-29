import numpy as np 
from data_process import *
from pca_model import *


def main():
    #load dataset
    x_train, y_train, x_test, y_test = load_data('G:\\DataSet\\att_faces')
    x_train, x_mean, eigVects = PCA().transform(x_train, K=50)
    
    x_test = np.array(np.dot((x_test - np.tile(x_mean, (np.shape(x_test)[0], 1))), eigVects))

    y_predict = []
    for index in x_test:
        tmp = np.array(x_train - np.tile(index, (np.shape(x_train)[0], 1)))
        tmp = np.sum(tmp**2, axis=1)
        y_predict.append(y_train[tmp.argmin()])
    y_predict = np.array(y_predict).T
    print("Accuracy %.2f%%"%((y_predict == y_test).mean()*100))

if __name__ == '__main__':
    main()