import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


class NeuralNetwork:

    def __init__(self):
        self.x = 1

    @staticmethod
    def sigmoid_unipolar_function(x: np.ndarray) -> np.ndarray:
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_bipolar_function(x: np.ndarray) -> np.ndarray:
        return (1-np.exp(-x))/(1 + np.exp(-x))
