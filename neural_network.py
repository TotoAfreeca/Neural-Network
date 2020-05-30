import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import random


class NeuralNetwork:

    def __init__(self):
        self.layers = []

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
                error = y_train[j] - output
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # calculate average error on all samples
            err /= samples
            print('epoch %d/%d   error=%f' % (i + 1, epochs, err))



    @staticmethod
    def sigmoid_bipolar_function(self, x):
        return (1-np.exp(-x))/(1 + np.exp(-x))

    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2))

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size

    def tanh(x):
        return np.tanh(x)

    def tanh_prime(x):
        return 1 - np.tanh(x) ** 2

    def sigmoid(x):
        return 1 / (1 + np.exp(-x))  # activation function

    def sigmoid_prime(x):
        return x * (1 - x)  # derivative of sigmoid
