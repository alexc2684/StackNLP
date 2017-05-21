import numpy as np
from rnn import RNN

X_train = np.load("X_train.npy")
y_train = np.load("y_train.npy")
X_train.reshape((X_train.shape[0], 1))
y_train.reshape((y_train.shape[0], 1))
np.random.seed(10)
rnn = RNN(10000)
o, s = rnn.forward_prop(X_train[10])
predictions = []
for i in range(10):
    out = rnn.predict(X_train[i])
    predictions.append(out)
# print(predictions)
print(rnn.loss([y_train[10], y_train[10]], [o, o]))
