import numpy as np
from Layer import Layer
import pandas as pd
import matplotlib.pyplot as plt
import random


class NeuralNetwork:

    def __init__(self):
        self.layers = []

    def create_layers(self, feature_size, output_size, layers_sizes, activation, activation_prime):

            if len(layers_sizes) == 0:
                self.add_layer(Layer(feature_size, output_size, activation, activation_prime))
            elif len(layers_sizes) == 1:
                self.add_layer(Layer(feature_size, layers_sizes[0], activation, activation_prime))
                self.add_layer(Layer(layers_sizes[0], output_size, activation, activation_prime))
            else:
                self.add_layer(Layer(feature_size, layers_sizes[0], activation, activation_prime))
                for i in range(1, len(layers_sizes)-1):
                    self.add_layer(Layer(layers_sizes[i - 1], layers_sizes[i], activation, activation_prime))
                self.add_layer(Layer(layers_sizes[len(layers_sizes) - 1], output_size, activation, activation_prime))

            return None

    def add_layer(self, layer):
        self.layers.append(layer)

    def predict(self, input_data):
        samples = len(input_data)
        result = []

        for i in range(samples):
            # forward prop
            output = input_data[i]
            for layer in self.layers:
                output = layer.forward_propagation(output)
            result.append(output)

        return result

    def train(self, x_train, y_train, epochs, learning_rate):
        samples = len(x_train)
        # training loop
        for i in range(epochs):
            err = 0
            for j in range(samples):
                # forward propagation
                output = x_train[j]
                for layer in self.layers:
                    output = layer.forward_propagation(output)
                #print(f'Output = {output}, desired = {y_train[j]}')
                # compute loss (for display purpose only)
                err += self.mse(y_train[j], output)

                # backward propagation
                error = self.mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))



    def sigmoid_bipolar_function(self, x):
        return (1-np.exp(-x))/(1 + np.exp(-x))

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2));

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size;
    def tanh(x):
        return np.tanh(x)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    def sigmoid_unipolar_function(self, x):
        "Numerically stable sigmoid function."
        if x >= 0:
            z = np.exp(-x)
            return 1 / (1 + z)
        else:
            # if x is less than zero then z will be small, denom can't be
            # zero because it's 1+z.
            z = np.exp(x)
            return z / (1 + z)

    def sigmoid_unipolar_prime(self, z):
        return self.sigmoid_unipolar_function(z) * (1 - self.sigmoid_unipolar_function(z))


