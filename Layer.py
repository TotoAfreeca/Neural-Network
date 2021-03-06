import numpy as np


# A class that emulates the behaviour of a simple layer of neurons
class Layer():

    def __init__(self, input_size, output_size, activation, activation_prime):
        self.input_size = input_size
        self.output_size = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5
        self.activation = activation
        self.activation_prime = activation_prime

    #feed forwards the data
    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(np.dot(self.input, self.weights) + self.bias)
        return self.output

    #backprop
    def back_propagation(self, output_error, learning_rate):

        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)
        # dBias = output_error

        # update parameters
        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error


        return self.activation_prime(self.input) * input_error

    def initialize_random_weights(self):
        self.weights = np.random.rand(self.input_size, self.output_size) - 0.5
        self.bias = np.random.rand(1, self.output_size) - 0.5
