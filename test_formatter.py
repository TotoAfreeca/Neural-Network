from DataFormatter import DataFormatter
import numpy as np
from Layer import Layer
from NeuralNetwork import NeuralNetwork


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


formatter = DataFormatter('iris.csv')

x_train, y_train = formatter.get_training_set()
x_test, y_test = formatter.get_test_set()
input_size, output_size = formatter.get_sizes()




# training data


# network
net = NeuralNetwork()
net.add_layer(Layer(input_size, 4, sigmoid_unipolar_function, sigmoid_unipolar_prime))
net.add_layer(Layer(4, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime))
net.add_layer(Layer(3, 4, sigmoid_unipolar_function, sigmoid_unipolar_prime))
net.add_layer(Layer(4, output_size, sigmoid_unipolar_function, sigmoid_unipolar_prime))

net.train(x_train, y_train, epochs=10000, learning_rate=0.02)



