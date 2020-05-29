import numpy as np

class Layer:

    def __init__(self, input_size, output_size, activation_function, activation_prime):

        self.input = input_size
        self.output = output_size
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.random.rand(1, output_size) - 0.5

        self.activation_function = activation_function
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation_function(np.dot(self.input, self.weights) + self.bias)

    def backward_propagation(self, output_error, learning_rate):
        input_error = np.dot(output_error, self.weights.T)
        weights_error = np.dot(self.input.T, output_error)

        self.weights -= learning_rate * weights_error
        self.bias -= learning_rate * output_error

        return self.activation_prime()