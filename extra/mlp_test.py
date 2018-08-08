from scipy.io import loadmat
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle
import matplotlib.pyplot as plt
import random

def show_digit(X, y_actual, y_pred):
    some_digit_image = X.reshape(28, 28)

    plt.imshow(
        some_digit_image, 
        cmap = plt.cm.binary,
        interpolation="nearest")
    tmp = "actual = " + str(int(y_actual)) + ", pred = " + str(int(y_pred))
    plt.title(tmp)
    plt.axis("off")
    plt.show()

mnist_raw = loadmat("mldata/mnist-original.mat")
mnist = {
    "data": mnist_raw["data"].T,
    "target": mnist_raw["label"][0],
    "COL_NAMES": ["label", "data"],
    "DESCR": "mldata.org dataset: mnist-original",
    }

X,y = mnist['data'], mnist['target']

shuffle_index = np.random.permutation(70000)
X,y = X[shuffle_index], y[shuffle_index]

X_train, X_test, y_train, y_test = X[:60000], X[60000:], y[:60000], y[60000:]

# load the model from disk
filename = 'mlp_digit.model'
mlp = pickle.load(open(filename, 'rb'))

for i in range(0,2):
	test_digit = int(random.random() * 5000)
	yfit = mlp.predict([X_test[test_digit]])

	print("actual digit", int(y_test[test_digit]))
	print("predict digit", int(yfit))
	show_digit(X_test[test_digit], y_test[test_digit], yfit)
