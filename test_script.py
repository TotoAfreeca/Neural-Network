import numpy as np
from Layer import Layer
from neural_network import NeuralNetwork


def sigmoid_unipolar_function(x):
    pos_mask = (x >= 0)
    neg_mask = (x < 0)
    z = np.zeros_like(x)
    z[pos_mask] = np.exp(-x[pos_mask])
    z[neg_mask] = np.exp(x[neg_mask])
    top = np.ones_like(x)
    top[neg_mask] = z[neg_mask]
    return top / (1 + z)


def sigmoid_unipolar_prime(z):
    return sigmoid_unipolar_function(z) * (1 - sigmoid_unipolar_function(z))

def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2


# training data

x_train = np.array([[[0, 0, 1]], [[0, 1, 1]], [[1, 0, 1]], [[0, 1, 0]], [[1, 0, 0]], [[1, 1, 1]], [[0, 0, 0]]])
y_train = np.array([0, 1, 1, 1, 1, 0, 0]).T

# network
net = NeuralNetwork()
net.add_layer(Layer(3, 4, sigmoid_unipolar_function, sigmoid_unipolar_prime))
net.add_layer(Layer(4, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime))

net.train(x_train, y_train, epochs=20000, learning_rate=0.01)

# test
out = net.predict([[1,0,1]])
print(out)


