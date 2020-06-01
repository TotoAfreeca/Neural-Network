import numpy as np

class Layer():

    def __init__(self, input_size, output_size, activation, activation_prime):
        self.weights = np.random.rand(input_size, output_size) - 0.5
        self.bias = np.ones(output_size)
        self.activation = activation
        self.activation_prime = activation_prime

    def forward_propagation(self, input_data):
        self.input = input_data
        self.output = self.activation(np.dot(self.input, self.weights) + self.bias)
        return self.output

    def back_propagation(self, output_error, learning_rate):

        delta = output_error * self.activation_prime(self.output)

        layer_adjustment = self.input.T.dot(delta)

        self.weights += learning_rate * layer_adjustment
        #self.bias += learning_rate * layer_adjustment

        return delta.dot(self.weights.T)