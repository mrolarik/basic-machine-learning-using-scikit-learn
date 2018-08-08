from scipy.io import loadmat
import numpy as np
from sklearn.neural_network import MLPClassifier
import pickle

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

# Neural Network
mlp = MLPClassifier(hidden_layer_sizes=(100,), max_iter=100, alpha=1e-4,
                    solver='sgd', verbose=10, tol=1e-5, random_state=1,
                    learning_rate_init=0.001)

mlp.fit(X_train, y_train)

# save the model to disk
filename = 'mlp_digit.model'
pickle.dump(mlp, open(filename, 'wb'))