import numpy as np

def sigmoid(z):
    return 1 / (1 + np.exp(-z))

def sigmoid_prime(z):
    return sigmoid(z) * (1 - sigmoid(z))

def softmax(z):
    z = z - np.max(z)
    ex = np.exp(z)
    return ex / np.sum(ex)

def softmax_prime(z):
    return np.ones(z.shape)

def relu(z):
    return np.maximum(0, z)

def relu_prime(z):
    return (z > 0).astype(float)

def tanh_prime(z):
    return 1.0 - np.tanh(z) ** 2

'''
Activation

compute(X): return the activation function applied element wise
compute_prime(X): return the derivate of the activation function applied element wise

'''

class SigmoidActivation:

    def __init__(self):
        self.name = 'sigmoid'

    def compute(self, x):
        return  sigmoid(x)

    def compute_prime(self, x):
        return  sigmoid_prime(x)


class SoftmaxActivation:

    def __init__(self):
        self.name = 'softmax'

    def compute(self, x):
        return softmax(x)

    def compute_prime(self, x):
        return softmax_prime(x)


class ReluActivation:

    def __init__(self):
        self.name = 'relu'

    def compute(self, x):
        return  relu(x)

    def compute_prime(self, x):
        return  relu_prime(x)

class TanhActivation:

    def __init__(self):
        self.name = 'tanh'

    def compute(self, x):
        return  np.tanh(x)

    def compute_prime(self, x):
        return  tanh_prime(x)

class LinearActivation:

    def __init__(self):
        self.name = 'linear'

    def compute(self, x):
        return x

    def compute_prime(self, x):
        return np.ones(x.shape)
