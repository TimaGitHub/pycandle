import numpy as np
from pycandle.parameter import Parameter

class Sigmoid:

    def __init__(self):
        self.input = None
        Parameter([self, []])

    @staticmethod
    def sigmoid_(x):
        return 1 / (1 + np.exp(-x))

    def __call__(self, x):
        self.input = x + 0
        return self.sigmoid_(x)

    def derivative(self):
        return (1 - self.sigmoid_(self.input)) * self.sigmoid_(self.input)


class Relu:

    def __init__(self):
        self.input = None
        Parameter([self, []])

    def __call__(self, x):
        self.input = x + 0
        return x * (1 + np.sign(x)) / 2

    def derivative(self):
        return (1 + np.sign(self.input)) / 2



class Leaky_relu:

    def __init__(self, a = 0.2):
        self.input = None
        self.a = a
        Parameter([self, []])

    def __call__(self, x):
        self.input = x + 0
        return x * ((1 + np.sign(x)) / 2 + self.a * (1 + np.sign(-x)) / 2)

    def derivative(self):
        return ((1 + np.sign(self.input)) / 2 + self.a * (1 + np.sign(-self.input)) / 2)

class Tanh:

    def __init__(self):
        self.input = None
        Parameter([self, []])

    @staticmethod
    def tanh_(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    def __call__(self, x):
        self.input = x + 0
        return self.tanh_(x)

    def derivative(self):
        return 1 - self.tanh_(self.input)


