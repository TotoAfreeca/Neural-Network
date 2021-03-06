import numpy as np
from Layer import Layer


class NeuralNetwork:

    def __init__(self):
        self.layers = []
        self.err = 0

    def add_layer(self, layer):
        self.layers.append(layer)


    #creates layers of the given input, output and layer sizes
    def create_layers(self, feature_size, output_size, layers_sizes, activation, activation_prime):

        if len(layers_sizes) > 0:
            self.add_layer(Layer(feature_size, layers_sizes[0], activation, activation_prime))
            for i in range(0, len(layers_sizes)-1):
                self.add_layer(Layer(layers_sizes[i], layers_sizes[i+1], activation, activation_prime))
            self.add_layer(Layer(layers_sizes[-1], output_size, activation, activation_prime))
        else:
            self.add_layer(Layer(feature_size, output_size, activation, activation_prime))

    def randomize_layers(self):
        for layer in self.layers:
            layer.initialize_random_weights()

    #predicts values for the input data - essentially feedforwards the net each time
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

                # compute loss (for display purpose only)
                err += self.mse(y_train[j], output)

                # backward propagation
                error = self.mse_prime(y_train[j], output)
                for layer in reversed(self.layers):
                    error = layer.back_propagation(error, learning_rate)

            # calculate average error on all samples, sets in in the field
            err /= samples
            #print("Epoch: " + str(i) + " error: "+ str(err))
            self.err = err

    #calculates mse over given set
    def calculate_test_mse(self, x_test, y_test):
        test_err = 0
        for i in range(len(x_test)):
            prediction = self.predict(x_test[i])
            test_err += self.mse(y_test[i], prediction)

        test_err /= len(x_test)
        return test_err



    def mse(self, y_true, y_pred):
        return np.mean(np.power(y_true - y_pred, 2));

    def mse_prime(self, y_true, y_pred):
        return 2 * (y_pred - y_true) / y_true.size;





