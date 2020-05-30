import numpy as np
from keras.datasets import mnist
from neural_network import NeuralNetwork
from HiddenLayer import HiddenLayer
from keras.utils import np_utils



def sigmoid_unipolar_function(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_unipolar_prime(z):
    return sigmoid_unipolar_function(z) * (1 - sigmoid_unipolar_function(z))


def tanh(x):
    return np.tanh(x)


def tanh_prime(x):
    return 1-np.tanh(x)**2

np.set_printoptions(suppress=True)

# load MNIST from server
(x_train, y_train), (x_test, y_test) = mnist.load_data()

# training data : 60000 samples
# reshape and normalize input data
x_train = x_train.reshape(x_train.shape[0], 1, 28*28)
x_train = x_train.astype('float32')
x_train /= 255
# encode output which is a number in range [0,9] into a vector of size 10
# e.g. number 3 will become [0, 0, 0, 1, 0, 0, 0, 0, 0, 0]
y_train = np_utils.to_categorical(y_train)

# same for test data : 10000 samples
x_test = x_test.reshape(x_test.shape[0], 1, 28*28)
x_test = x_test.astype('float32')
x_test /= 255
y_test = np_utils.to_categorical(y_test)

# Network
net = NeuralNetwork()
net.add_layer(HiddenLayer(28*28, 100, sigmoid_unipolar_function, sigmoid_unipolar_prime ))                # input_shape=(1, 28*28)    ;   output_shape=(1, 100)
net.add_layer(HiddenLayer(100, 50, sigmoid_unipolar_function, sigmoid_unipolar_prime))                   # input_shape=(1, 100)      ;   output_shape=(1, 50)
net.add_layer(HiddenLayer(50, 10, sigmoid_unipolar_function, sigmoid_unipolar_prime))                    # input_shape=(1, 50)       ;   output_shape=(1, 10)

# train on 1000 samples
# as we didn't implemented mini-batch GD, training will be pretty slow if we update at each iteration on 60000 samples...
net.train(x_train[0:1000], y_train[0:1000], epochs=35, learning_rate=0.1)

# test on 3 samples
out = net.predict(x_test[0:3])
print("\n")
print("predicted values : ")
print(out, end="\n")
print("true values : ")
print(y_test[0:3])
