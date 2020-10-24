from unittest import TestCase
from NeuralNetwork import NeuralNetwork
from Layer import Layer
from Functions import sigmoid_unipolar_function, sigmoid_unipolar_prime
import numpy as np

# Basic unit test to work on automatically builded network


class TestNeuralNetwork(TestCase):
    def setUp(self):
        self.neural_network = NeuralNetwork()

    def test_create_layers_multi(self):
        self.neural_network.create_layers(3, 3, [4, 3], sigmoid_unipolar_function, sigmoid_unipolar_prime )

        neural = NeuralNetwork()

        layer1 = Layer(3, 4, sigmoid_unipolar_function, sigmoid_unipolar_prime)
        layer2 = Layer(4, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime)
        layer3 = Layer(3, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime)


        shapes1 = []
        shapes2 = [np.shape(layer1.weights), np.shape(layer2.weights), np.shape(layer3.weights)]
        for layer in self.neural_network.layers:
            shapes1.append(np.shape(layer.weights))
        print("Function shapes: " + str(shapes1))
        print("Created shapes: " + str(shapes2))

        self.assertEqual(shapes1, shapes2)

    def test_create_layers_no_hidden(self):
        self.neural_network.create_layers(3, 3, [], sigmoid_unipolar_function,
                                          sigmoid_unipolar_prime)

        neural = NeuralNetwork()

        layer1 = Layer(3, 3, sigmoid_unipolar_function, sigmoid_unipolar_prime)

        shapes1 = []
        shapes2 = [np.shape(layer1.weights)]
        for layer in self.neural_network.layers:
            shapes1.append(np.shape(layer.weights))

        print("Function shapes: " + str(shapes1))
        print("Created shapes: " + str(shapes2))

        self.assertEqual(shapes1, shapes2)