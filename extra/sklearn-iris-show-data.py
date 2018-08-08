import numpy as np
from sklearn.datasets import load_iris

iris_dataset = load_iris()

x = iris_dataset.data   # x = iris_dataset['data']
y = iris_dataset.target	# y = iris_dataset['target']

len_y = len(y)
new_y = np.reshape(y,(len_y,1))
dataset = np.append(new_y,x, axis=1)

print(dataset)