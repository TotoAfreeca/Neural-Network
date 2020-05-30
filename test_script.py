import numpy as np
from HiddenLayer import HiddenLayer
from neural_network import NeuralNetwork

def sigmoid_unipolar_function(x):
    return 1 / (1 + np.exp(-x))

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
net.add_layer(HiddenLayer(3, 4, sigmoid_unipolar_function, sigmoid_unipolar_prime))
net.add_layer(HiddenLayer(4, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime))

net.train(x_train, y_train, epochs=3000, learning_rate=0.05)

# test
out = net.predict([[1,1,1]])
print(out)


