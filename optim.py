import numpy as np
import pycandle.nn

'''
note you can use here mm.Parameter for the latest Parameter object
and self.model._constructor_Parameter for the certain model
by default i use here model._constructor_Parameter
'''


class SGD:
    def __init__(self, model, learning_rate):
        self.model = model
        self.lr = learning_rate

    def step(self, loss):

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0], self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1] ]
            elif type(layer).__name__ == 'RNN':
                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0],
                                            self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1],
                                            self.model._constructor_Parameter.calling[layer][2] - self.lr * loss.backward_list[index][2],
                                            self.model._constructor_Parameter.calling[layer][3] - self.lr * loss.backward_list[index][3],
                                            self.model._constructor_Parameter.calling[layer][4] - self.lr * loss.backward_list[index][4]
                                            ]

class NAG:
    def __init__(self, model, learning_rate, momentum):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.last_grad_w = None
        self.last_grad_b = None

    def step(self, loss):

        if self.last_grad_w == None:
            self.last_grad_w = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_b = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_U = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_W = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_V = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_bh = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_by = [0] * len(self.model._constructor_Parameter.layers)

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                self.last_grad_w[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_w[index]
                self.last_grad_b[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_b[index]
                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_w[index], self.model._constructor_Parameter.calling[layer][1] + self.last_grad_b[index] ]

            elif type(layer).__name__ == 'RNN':
                self.last_grad_U[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_U[index]
                self.last_grad_W[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_W[index]
                self.last_grad_V[index] = - self.lr * loss.backward_list[index][2] + self.momentum * self.last_grad_V[index]
                self.last_grad_bh[index] = - self.lr * loss.backward_list[index][3] + self.momentum * self.last_grad_bh[index]
                self.last_grad_by[index] = - self.lr * loss.backward_list[index][4] + self.momentum * self.last_grad_by[index]

                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_U[index],
                                            self.model._constructor_Parameter.calling[layer][1] + self.last_grad_U[index],
                                            self.model._constructor_Parameter.calling[layer][2] + self.last_grad_U[index],
                                            self.model._constructor_Parameter.calling[layer][3] + self.last_grad_U[index],
                                            self.model._constructor_Parameter.calling[layer][4] + self.last_grad_U[index]
                                            ]

class RMS:
    def __init__(self, model, learning_rate, ro):

        self.model = model
        self.lr = learning_rate
        if ro < 0 or ro > 1:
            raise Exception("Incorrect ro value")
        self.ro = ro
        self.grad_velocity_w = None
        self.grad_velocity_b = None

    def step(self, loss):

        if self.grad_velocity_w== None:
            self.grad_velocity_w = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_b = [0] * len(self.model._constructor_Parameter.layers)

            self.grad_velocity_U = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_W = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_V = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_bh = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_by = [0] * len(self.model._constructor_Parameter.layers)

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                self.grad_velocity_w[index] = self.ro * self.grad_velocity_w[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
                self.grad_velocity_b[index] = self.ro * self.grad_velocity_b[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0] / np.sqrt(self.grad_velocity_w[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1] / np.sqrt(self.grad_velocity_b[index] + 1e-5) ]

            elif type(layer).__name__ == 'RNN':
                self.grad_velocity_U[index] =  self.ro * self.grad_velocity_U[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
                self.grad_velocity_W[index] =  self.ro * self.grad_velocity_W[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
                self.grad_velocity_V[index] =  self.ro * self.grad_velocity_V[index] + (1 - self.ro) * loss.backward_list[index][2] ** 2
                self.grad_velocity_bh[index] =  self.ro * self.grad_velocity_bh[index] + (1 - self.ro) * loss.backward_list[index][3] ** 2
                self.grad_velocity_by[index] =  self.ro * self.grad_velocity_by[index] + (1 - self.ro) * loss.backward_list[index][4] ** 2

                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] - self.lr * loss.backward_list[index][0] / np.sqrt(self.grad_velocity_U[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][1] - self.lr * loss.backward_list[index][1] / np.sqrt(self.grad_velocity_W[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][2] - self.lr * loss.backward_list[index][2] / np.sqrt(self.grad_velocity_V[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][3] - self.lr * loss.backward_list[index][3] / np.sqrt(self.grad_velocity_bh[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][4] - self.lr * loss.backward_list[index][4] / np.sqrt(self.grad_velocity_wby[index] + 1e-5),
                                            ]

class ADAM:
    def __init__(self, model, learning_rate, momentum, ro):
        self.model = model
        self.lr = learning_rate
        self.momentum = momentum
        self.last_grad_w = None
        self.last_grad_b = None

        if ro < 0 or ro > 1:
            raise Exception("Incorrect ro value")

        self.ro = ro
        self.grad_velocity_w = None
        self.grad_velocity_b = None


    def step(self, loss):

        if self.last_grad_w == None:
            self.last_grad_w = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_b = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_U = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_W = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_V = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_bh = [0] * len(self.model._constructor_Parameter.layers)
            self.last_grad_by = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_w = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_b = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_U = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_W = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_V = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_bh = [0] * len(self.model._constructor_Parameter.layers)
            self.grad_velocity_by = [0] * len(self.model._constructor_Parameter.layers)

        for index, layer in enumerate(self.model._constructor_Parameter.layers[::-1]):
            if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                self.grad_velocity_w[index] = self.ro * self.grad_velocity_w[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
                self.grad_velocity_b[index] = self.ro * self.grad_velocity_b[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
                self.last_grad_w[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_w[index]
                self.last_grad_b[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_b[index]
                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_w[index] / np.sqrt(self.grad_velocity_w[index] + 1e-5), self.model._constructor_Parameter.calling[layer][1] + self.last_grad_b[index] / np.sqrt(self.grad_velocity_b[index] + 1e-5)]

            elif type(layer).__name__ == 'RNN':
                self.grad_velocity_U[index] = self.ro * self.grad_velocity_U[index] + (1 - self.ro) * loss.backward_list[index][0] ** 2
                self.grad_velocity_W[index] = self.ro * self.grad_velocity_W[index] + (1 - self.ro) * loss.backward_list[index][1] ** 2
                self.grad_velocity_V[index] = self.ro * self.grad_velocity_V[index] + (1 - self.ro) * loss.backward_list[index][2] ** 2
                self.grad_velocity_bh[index] = self.ro * self.grad_velocity_bh[index] + (1 - self.ro) * loss.backward_list[index][3] ** 2
                self.grad_velocity_by[index] = self.ro * self.grad_velocity_by[index] + (1 - self.ro) * loss.backward_list[index][4] ** 2
                self.last_grad_U[index] = - self.lr * loss.backward_list[index][0] + self.momentum * self.last_grad_U[index]
                self.last_grad_W[index] = - self.lr * loss.backward_list[index][1] + self.momentum * self.last_grad_W[index]
                self.last_grad_V[index] = - self.lr * loss.backward_list[index][2] + self.momentum * self.last_grad_V[index]
                self.last_grad_bh[index] = - self.lr * loss.backward_list[index][3] + self.momentum * self.last_grad_bh[index]
                self.last_grad_by[index] = - self.lr * loss.backward_list[index][4] + self.momentum * self.last_grad_by[index]

                self.model._constructor_Parameter.calling[layer] = [self.model._constructor_Parameter.calling[layer][0] + self.last_grad_U[index] / np.sqrt(self.grad_velocity_U[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][1] + self.last_grad_W[index] / np.sqrt(self.grad_velocity_W[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][2] + self.last_grad_V[index] / np.sqrt(self.grad_velocity_V[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][3] + self.last_grad_bh[index] / np.sqrt(self.grad_velocity_bh[index] + 1e-5),
                                            self.model._constructor_Parameter.calling[layer][4] + self.last_grad_by[index] / np.sqrt(self.grad_velocity_by[index] + 1e-5),
                                            ]
