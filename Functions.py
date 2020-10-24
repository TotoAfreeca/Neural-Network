#File to maintain all the activation functions and their derivatives

import numpy as np


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
    return 1 - np.tanh(x) ** 2
