import numpy as np

import scipy.signal
from pycandle.parameter import ParameterObj


Parameter = None

class Module:

    def __init__(self):
        self._constructor_Parameter = ParameterObj()
        global Parameter
        Parameter = self._constructor_Parameter

    def forward(self):
        pass

    def __call__(self, x):
        global Parameter
        Parameter = self._constructor_Parameter
        if x.ndim == 1:
            return self.forward(x.reshape(1, -1))
        else:
            return self.forward(x)

    def parameters(self, show_all = False):
        for layer in self._constructor_Parameter.layers:
            if type(layer).__name__ == 'Linear':
                if show_all:
                    print('Linear:', "weights:", self._constructor_Parameter.calling[layer][0].shape, ", bias:", self._constructor_Parameter.calling[layer][1].shape,  *self._constructor_Parameter.calling[layer])
                else:
                    print('Linear:', "weights:", self._constructor_Parameter.calling[layer][0].shape, ", bias:", self._constructor_Parameter.calling[layer][1].shape)

            elif type(layer).__name__ == 'Conv':
                if show_all:
                    print('Conv:')
                    for index, element in enumerate(self._constructor_Parameter.calling[layer][0]):
                        print(f"kernel №{index + 1}", element)
                    print("bias:", self._constructor_Parameter.calling[layer][1])
                else:
                    print("Conv:", self._constructor_Parameter.calling[layer][0][0].shape, ", {} kernels ,".format(len(self._constructor_Parameter.calling[layer][0])), "bias:", self._constructor_Parameter.calling[layer][1].shape)


            elif type(layer).__name__ == 'BatchNorm':
                if show_all:
                    print('BatchNorm:', ", gamma: ", f'({len(self._constructor_Parameter.calling[layer][0])},)', self._constructor_Parameterrameter.calling[layer][0], ", betta: ", f'({len(self._constructor_Parameter.calling[layer][0])},)', self._constructor_Parameter.calling[layer][1])
                else:
                    print("BatchNorm:", len(self._constructor_Parameter.calling[layer][0]))


    def save(self, path='model_params.npy'):
        with open(path, 'wb') as f:
            for layer in Parameter.layers:
                if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                    np.save(f, Parameter.calling[layer][0])
                    np.save(f, Parameter.calling[layer][1])

    def load(self, path='model_params.npy'):

        with open(path, 'rb') as f:
            for layer in Parameter.layers:
                if type(layer).__name__ in ('Linear', 'Conv', 'BatchNorm'):
                    Parameter.calling[layer][0] = np.load(f)
                    Parameter.calling[layer][1] = np.load(f)

class Linear:
    def __init__(self, input_channels: int, output_channels: int, bias = True):
        if (isinstance(input_channels, int) & isinstance(output_channels, int)) == False:
            raise Exception("Incorrect linear layer initialization")

        self.input_channels = input_channels
        self.output_channels = output_channels
        self.bias = bias
        if bias:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.input_channels, self.output_channels)), np.random.uniform(- 0.5, 0.5, size=self.output_channels)])
        else:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.input_channels, self.output_channels)), np.zeros(self.output_channels)])


    def __call__(self, x):
        self.hidden_output_no_activation = x + 0
        result = x @ Parameter.calling[self][0] + Parameter.calling[self][1]
        self.hidden_output_activation = result + 0
        return result

class Flatten:

    def __init__(self):
        self.width = None
        self.size = None
        Parameter([self, []])

    def __call__(self, x):

        self.width = x.shape[1]
        self.size = x.shape[2]
        return x.reshape(x.shape[0], -1)

class Conv:

    def __init__(self, input_channels: int, output_channels: int, kernel_size: tuple, bias = True):
        if (isinstance(input_channels, int) & isinstance(output_channels, int) & isinstance(kernel_size, (tuple, list)) & (len(kernel_size) == 2)) == False:
            raise Exception("Incorrect convolution layer initialization")
        self.bias = bias
        self.input_channels = input_channels
        self.kernel_size = (input_channels, kernel_size[0], kernel_size[1])
        self.n_filters = output_channels

        self.filter_array = np.array(
            [np.random.uniform(-1, 1, (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))])
        for i in range(1, self.n_filters):
            self.filter_array = np.append(self.filter_array, [
                np.random.uniform(-1, 1, (self.kernel_size[0], self.kernel_size[1], self.kernel_size[2]))], axis=0)
        if self.bias:
            Parameter([self, self.filter_array, np.random.uniform(-1, 1, (self.n_filters))])
        else:
            Parameter([self, self.filter_array, np.zeros(self.n_filters)])


    def __call__(self, x):
        if x.ndim == 3 and x.shape[0] == 1:
            x = x.reshape(1, x.shape[0], x.shape[1], x.shape[2])
        elif x.ndim == 3 and (not x.shape[0] == 1):
            x = x.reshape(x.shape[0], 1, x.shape[1], x.shape[2])
        elif x.ndim == 2 and x.shape[0] == x.shape[1]:
            x = x.reshape(1, 1, x.shape[0], x.shape[1])
        elif x.ndim == 2 and x.shape[0] != x.shape[1]:
            x = x.reshape(x.shape[0], 1, int(np.sqrt(x.shape[1])), int(np.sqrt(x.shape[1])))
        elif x.ndim > 4 or x.ndim == 1:
            raise Exception("Something wrong with input data into convolution layer")

        self.image = x.copy()
        new_image_array = np.zeros((x.shape[0], self.n_filters, x.shape[2] - self.kernel_size[1] + 1, x.shape[3] - self.kernel_size[2] + 1))
        for i in range(x.shape[0]):
            for j in range(self.n_filters):
                new_image_array[i][j] = np.squeeze(scipy.signal.fftconvolve(x[i], Parameter.calling[self][0][j], mode='valid'), axis=0) + Parameter.calling[self][1][j]
        return new_image_array


class RNN:
    '''
    note:
    E - input's dimension of one sample ( vector of features ) E - from "Embedding"
    H - input's dimension of the vector of Hidden state
    N - output's dimension of the vector
    B - batch size
    T - the length of the sequence (number of time periods (seconds) )
    more here: https://qudata.com/ml/ru/NN_RNN_Torch.html#LSTM
    '''

    def __init__(self, E: int, H: int, N: int, nonlinearity='tanh',  bias = True, batch_first=True, last_input: int = 0):

        if (isinstance(E, int) & isinstance(H, int) & isinstance(N, int)) == False:
            raise Exception("Incorrect reccurent layer initialization")

        self.E = E
        self.H = H
        self.N = N
        self.bias = bias
        self.nonlinearity = nonlinearity
        self.batch_first = batch_first
        self.last_input = last_input

        if bias:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.H, self.E)), np.random.uniform(- 0.5, 0.5, size=(self.H, self.H)), np.random.uniform(- 0.5, 0.5, size=(self.N, self.H)),  np.random.uniform(- 0.5, 0.5, size=self.H), np.random.uniform(- 0.5, 0.5, size=self.N)])
        else:
            Parameter([self, np.random.uniform(- 0.5, 0.5, size=(self.H, self.E)), np.random.uniform(- 0.5, 0.5, size=(self.H, self.H)), np.random.uniform(- 0.5, 0.5, size=(self.N, self.H)),  np.zeros(self.H), np.zeros(self.N)])

    @staticmethod
    def tanh_(x):
        return (np.exp(2 * x) - 1) / (np.exp(2 * x) + 1)

    @staticmethod
    def derivative_t(x):
        return 1 - RNN.tanh_(x) ** 2

    @staticmethod
    def relu_(x):
        return x * (1 + np.sign(x)) / 2

    @staticmethod
    def derivative_r(x):
        return (1 + np.sign(x)) / 2

    def __call__(self, x, h0 = None):

        '''
        if "batch_first=True":
            x shape: (B, T, E)
            result shape: (B, T, N)
        else:
            x shape: (T, B, E)
            result shape: (T, B, N)

        self.input shape: (T, E, B)
        self.output shape (T, N, B)
        h0 shape: (H, B)
        '''

        self.mask = np.zeros((x.shape[1], self.N))
        self.mask[-self.last_input:] = 1

        if self.batch_first:
            self.B = x.shape[0]
            self.T = x.shape[1]
            if h0 == None:
                self.h0 = np.zeros((self.H, self.B))
            self.input = np.transpose(x, axes=[1, 2, 0])

        else:
            self.B = x.shape[1]
            self.T = x.shape[0]
            if h0 == None:
                self.h0 = np.zeros((self.H, self.B))
            self.input = np.transpose(x, axes=[0, 2, 1])

        if self.nonlinearity == 'tanh':
            self.func = RNN.tanh_
            self.derivative = RNN.derivative_t
        else:
            self.func = RNN.relu_
            self.derivative = RNN.derivative_r

        self.hidden_no_activation = np.zeros(((self.T, self.H, self.B)))
        self.hidden_activation = np.zeros(((self.T, self.H, self.B)))

        if self.last_input == 0:
            self.last_input = self.T
        self.output_no_activation = np.zeros(((self.last_input, self.N, self.B)))
        self.output_activation = np.zeros(((self.last_input, self.N, self.B)))

        temp = self.h0 + 0
        temp_ = 0
        for t in range(self.T):
            temp_ = Parameter.calling[self][0] @ self.input[t] + Parameter.calling[self][1] @ temp + Parameter.calling[self][3].reshape(-1, 1)
            self.hidden_no_activation[t] = temp_ + 0
            temp = self.func(temp_)
            self.hidden_activation[t] = temp + 0

            if (self.last_input == 0):

                temp_ = Parameter.calling[self][2] @ temp + Parameter.calling[self][4].reshape(-1, 1)
                self.output_no_activation[t] = temp_ + 0
                temp__ = self.func(temp_)
                self.output_activation[t] = temp__ + 0

            elif (t >= self.T - self.last_input):
                temp_ = Parameter.calling[self][2] @ temp + Parameter.calling[self][4].reshape(-1, 1)
                self.output_no_activation[t - (self.T - self.last_input)] = temp_ + 0
                temp__ = self.func(temp_)
                self.output_activation[t - (self.T - self.last_input)] = temp__ + 0

        if self.batch_first:
            return np.transpose(self.output_activation, axes=[2, 0, 1])
        else:
            return np.transpose(self.output_activation, axes=[0, 2, 1])


class MaxPool:

    def __init__(self, kernel_size: tuple):
        if (isinstance(kernel_size, (tuple, list)) & (len(kernel_size) == 2)) == False:
            raise Exception("Incorrect maxpool layer initialization")

        self.kernel_size = kernel_size
        Parameter([self, []])

    def __call__(self, x):
        if x.shape[2] % self.kernel_size[0] != 0:
            raise Exception("Can't apply pooling due to the size, please change it")
        array = x.copy()
        result_full = np.zeros((array.shape[0], array.shape[1], int(array.shape[2] / self.kernel_size[0]), int(array.shape[3] / self.kernel_size[1])))

        for k in range(array.shape[0]):
            for m in range(array.shape[1]):
                result = []
                self.i = 0
                while self.i < array[k][m].shape[0] - self.kernel_size[0] + 1:
                    self.j = 0
                    while self.j < array[k][m].shape[1] - self.kernel_size[1] + 1:
                        result.append(np.max(array[k][m][self.i:self.i + self.kernel_size[0], self.j:self.j + self.kernel_size[1]]))
                        array[k][m][self.i:self.i + self.kernel_size[0], self.j:self.j + self.kernel_size[1]] = (array[k][m][self.i:self.i + self.kernel_size[0], self.j: self.j + self.kernel_size[1]]) * [array[k][m][self.i:self.i + self.kernel_size[0],
                                                                                                      self.j:self.j +self.kernel_size[1]] == np.max(array[k][m][self.i:self.i +self.kernel_size[0], self.j:self.j +self.kernel_size[1]])]

                        self.j += self.kernel_size[1]
                    self.i += self.kernel_size[0]

                result_full[k][m] = np.array(result).reshape(int(array[k][m].shape[0] / self.kernel_size[0]), int(array[k][m].shape[1] / self.kernel_size[1]))

        self.array = array
        return result_full

class MinPool:

    def __init__(self, kernel_size: tuple):
        if (isinstance(kernel_size, (tuple, list)) & (len(kernel_size) == 2)) == False:
            raise Exception("Incorrect minpool layer initialization")

        self.kernel_size = kernel_size

        Parameter([self, []])

    def __call__(self, x):
        if x.shape[2] % self.kernel_size[0] != 0:
            raise Exception("Can't apply pooling due to the size, please change it")
        array = x.copy()
        result_full = np.zeros((array.shape[0], array.shape[1], int(array.shape[2] / self.kernel_size[0]), int(array.shape[3] / self.kernel_size[1])))

        for k in range(array.shape[0]):
            for m in range(array.shape[1]):
                result = []
                self.i = 0
                while self.i < array[k][m].shape[0] - self.kernel_size[0] + 1:
                    self.j = 0
                    while self.j < array[k][m].shape[1] - self.kernel_size[1] + 1:
                        result.append(np.min(array[k][m][self.i:self.i + self.kernel_size[0], self.j:self.j + self.kernel_size[1]]))
                        array[k][m][self.i:self.i + self.kernel_size[0], self.j:self.j + self.kernel_size[1]] = (array[k][m][self.i:self.i + self.kernel_size[0],
                                                                                                                 self.j: self.j + self.kernel_size[1]]) * [
                                                                                                                    array[k][m][self.i:self.i + self.kernel_size[0],
                                                                                                                    self.j:self.j + self.kernel_size[1]] == np.min(
                                                                                                                        array[k][m][self.i:self.i + self.kernel_size[0],
                                                                                                                        self.j:self.j + self.kernel_size[1]])]

                        self.j += self.kernel_size[1]
                    self.i += self.kernel_size[0]

                result_full[k][m] = np.array(result).reshape(int(array[k][m].shape[0] / self.kernel_size[0]), int(array[k][m].shape[1] / self.kernel_size[1]))

        self.array = array
        return result_full

class DropOut:

    def __init__(self, q):
        if q < 0 or q > 1:
            raise Exception("Incorrect probability value")
        if type(Parameter.layers[-1]).__name__ != 'Linear':
            raise Exception("Please, use dropout only after a linear layer")
        self.q = q
        mask = np.random.choice([0, 1], Parameter.calling[Parameter.layers[-1]][0].shape[1], p = [q, 1 - q])
        Parameter.calling[Parameter.layers[-1]] = [Parameter.calling[Parameter.layers[-1]][0] * mask, Parameter.calling[Parameter.layers[-1]][1] * mask]
    def __call__(self, x):
        return x / self.q

class BatchNorm:

    def __init__(self, size):
        self.conv = False
        if type(Parameter.layers[-1]).__name__ == 'Conv':
            self.conv = True
        Parameter([self, np.ones((size)), np.ones((size))])

    def __call__(self, x):
        # for convolutional and linear layers batchnorm algorithm is different
        if self.conv:
            self.mean = np.mean(x, axis = (0, 2, 3))
            self.std = np.std(x, axis = (0, 2, 3)) + 0.0001
            self.x = np.zeros(x.shape)
            res = np.zeros(x.shape)
            for c in range(x.shape[1]):
                for i in range(x.shape[0]):
                    for j in range(x.shape[2]):
                        for k in range(x.shape[3]):
                            self.x[i, c, j, k] = (x[i, c, j, k] - self.mean[c]) / self.std[c]
                            res[i, c, j, k] = self.x[i, c, j, k] * Parameter.calling[self][0][c] + Parameter.calling[self][1][c]
            return res
        else:
            self.mean = np.mean(x, axis=(0))
            self.std = np.std(x, axis=(0)) + 0.0001
            self.x = (x - self.mean) / self.std
            return Parameter.calling[self][0] * self.x + Parameter.calling[self][1]

class CrossEntropyLoss:

    def __init__(self, l1_reg = 0, l2_reg = 0):
        self.backward_list = []
        self.predicted = None
        self.true = None
        self.l1_reg = l1_reg
        self.l2_reg = l2_reg

    def __call__(self, predicted, true):
        self.predicted = predicted + 0
        if predicted.ndim == 1:
            Parameter.number_of_classes = predicted.shape[0]
            self.true = np.int_(np.arange(0, Parameter.number_of_classes) == true)
            true = np.int_(np.arange(0, Parameter.number_of_classes) == true)
            self.loss = -1 * np.sum(true * np.log(predicted + 1e-5), axis=0)
            return self
        else:
            Parameter.number_of_classes = predicted.shape[1]
            self.true = np.int_(np.arange(0, Parameter.number_of_classes) == true)
            true = np.int_(np.arange(0, Parameter.number_of_classes) == true)
            self.loss = -1 * np.sum(true * np.log(predicted + 1e-5), axis=1)
            return self

    def backward(self):
        self.backward_list = []
        loss = self.predicted - self.true

        for index, layer in enumerate(Parameter.layers[::-1]):

            if np.isnan(loss).any():
                raise Exception("NAN values detected in loss. Please change network parameters")

            if type(layer).__name__ == 'Linear':

                changes_w = (layer.hidden_output_no_activation.T @ loss + self.l2_reg * Parameter.calling[layer][0] + self.l1_reg * np.sign(Parameter.calling[layer][0]) ) / loss.shape[0]

                if layer.bias:
                    changes_b = (np.sum(loss) / loss.shape[0])
                else:
                    changes_b = 0

                self.backward_list.append([changes_w, changes_b])
                loss = loss @ Parameter.calling[layer][0].T

            elif type(layer).__name__ == 'Flatten':

                if type(Parameter.layers[::-1][index + 1]).__name__ == 'RNN':
                    loss = loss.reshape(loss.shape[0], layer.width, layer.size)
                else:
                    loss = loss.reshape(loss.shape[0], layer.width, layer.size, layer.size)
                self.backward_list.append([])

            elif type(layer).__name__ == 'Conv':

                d_image = np.zeros(Parameter.calling[layer][0].shape)

                d_ = np.zeros(Parameter.calling[layer][1].shape)

                temp2 = np.zeros(layer.image.shape[0])
                temp = np.zeros((layer.image.shape[0], *layer.kernel_size[1:]))

                for i in range(layer.n_filters):
                    for j in range(layer.kernel_size[0]):
                        for k in range(layer.image.shape[0]):
                            temp[k] = scipy.signal.fftconvolve(layer.image[i][j], loss[k][i], mode='valid')
                        d_image[i][j] = temp.mean(axis=0)
                if layer.bias:
                    for i in range(layer.n_filters):
                        for j in range(layer.image.shape[0]):
                            temp2[j] = np.sum(loss[i])
                        d_[i] = temp2.mean()

                if index != len(Parameter.layers) - 1:
                    rot_filter_array = np.zeros(Parameter.calling[layer][0].shape)

                    for i in range(Parameter.calling[layer][0].shape[0]):
                        rot_filter_array[i] = np.rot90(np.rot90(Parameter.calling[layer][0][i], -1, (1, 2)), -1, (1, 2))

                    padded = np.pad(loss, ((0, 0), (0, 0), (layer.kernel_size[1] - 1, layer.kernel_size[1] - 1),
                                           (layer.kernel_size[1] - 1, layer.kernel_size[1] - 1)), 'constant', constant_values=(0))

                    new_loss = np.zeros(layer.image.shape)

                    temp = np.zeros((layer.n_filters, layer.image.shape[2], layer.image.shape[3]))

                    for i in range(padded.shape[0]):
                        for k in range(layer.image.shape[1]):
                            for j in range(rot_filter_array.shape[0]):
                                temp[j] = scipy.signal.fftconvolve(padded[i][j], rot_filter_array[j][k], mode='valid')
                        new_loss[i][k] = np.mean(temp, axis=0)

                    loss = new_loss + 0
                self.backward_list.append([d_image / loss.shape[0], d_ / loss.shape[0] ])


            elif type(layer).__name__ == 'MaxPool':
                new_shape = np.zeros(layer.array.shape)
                for k in range(layer.array.shape[0]):
                    for m in range(layer.array.shape[1]):
                        inx_ = 0
                        inx__ = 0
                        layer.i = 0
                        while layer.i < layer.array[k][m].shape[0] - layer.kernel_size[0] + 1:
                            layer.j = 0
                            inx__ = 0
                            while layer.j < layer.array[k][m].shape[1] - layer.kernel_size[1] + 1:
                                new_shape[k][m][layer.i:layer.i + layer.kernel_size[0], layer.j:layer.j + layer.kernel_size[1]] = \
                                    loss[k][m][inx_][inx__]
                                inx__ += 1
                                layer.j += layer.kernel_size[1]

                            inx_ += 1
                            layer.i += layer.kernel_size[0]

                loss = np.squeeze([layer.array > 0] * new_shape, axis=0)

                self.backward_list.append([])

            elif type(layer).__name__ == 'MinPool':
                new_shape = np.zeros(layer.array.shape)
                for k in range(layer.array.shape[0]):
                    for m in range(layer.array.shape[1]):
                        inx_ = 0
                        inx__ = 0
                        layer.i = 0
                        while layer.i < layer.array[k][m].shape[0] - layer.kernel_size[0] + 1:
                            layer.j = 0
                            inx__ = 0
                            while layer.j < layer.array[k][m].shape[1] - layer.kernel_size[1] + 1:
                                new_shape[k][m][layer.i:layer.i + layer.kernel_size[0], layer.j:layer.j + layer.kernel_size[1]] = \
                                    loss[k][m][inx_][inx__]
                                inx__ += 1
                                layer.j += layer.kernel_size[1]

                            inx_ += 1
                            layer.i += layer.kernel_size[0]

                loss = np.squeeze([layer.array > 0] * new_shape, axis=0)

                self.backward_list.append([])

            elif type(layer).__name__ == 'DropOut':
                loss = loss * layer.q
                self.backward_list.append([])

            elif type(layer).__name__ == 'BatchNorm':

                if type(Parameter.layers[::-1][index + 1]).__name__ == 'Conv':
                    self.backward_list.append([np.sum(loss * layer.x, axis=(0, 2, 3)), np.sum(loss, axis=(0, 2, 3))])


                    dl_dx = np.zeros(loss.shape)
                    dl_dstd = np.zeros(loss.shape)
                    dl_dmean = np.zeros(loss.shape)
                    for c in range(layer.x.shape[1]):
                        for i in range(layer.x.shape[0]):
                            for j in range(layer.x.shape[2]):
                                for k in range(layer.x.shape[3]):
                                    dl_dx[i, c, j, k] = loss[i, c, j, k] * Parameter.calling[layer][0][c]
                                    dl_dstd[i, c, j, k] = dl_dx[i, c, j, k] * (layer.x[i, c, j, k] * layer.std[c]) * (-1 / 2) / (layer.std[c] ** 3)
                                    dl_dmean[i, c, j, k] = dl_dx[i, c, j, k] * (- 1 / layer.std[c]) + dl_dstd[i, c, j, k] * (-2 * layer.x[i, c, j, k] * layer.std[c] / layer.x.shape[0] / layer.x.shape[2] / layer.x.shape[3])

                    dl_dstd = np.sum(dl_dstd, axis=(0, 2, 3))
                    dl_dmean = np.sum(dl_dmean, axis=(0, 2, 3))

                    for c in range(layer.x.shape[1]):
                        for i in range(layer.x.shape[0]):
                            for j in range(layer.x.shape[2]):
                                for k in range(layer.x.shape[3]):
                                    loss[i, c, j, k] = dl_dx[i, c, j, k] / layer.std[c] + dl_dstd[c] * 2 * (layer.x[i, c, j, k] * layer.std[c]) / layer.x.shape[0] / layer.x.shape[2] / layer.x.shape[3] + dl_dmean[c] / layer.x.shape[0] / layer.x.shape[2] / layer.x.shape[3]

                else:
                    self.backward_list.append([np.sum(loss * layer.x, axis = 0), np.sum(loss, axis = 0)])
                    dl_dx = loss * Parameter.calling[layer][0]
                    dl_dstd = np.sum(dl_dx * (layer.x * layer.std) * (-1/2) / (layer.std ** 3), axis = 0)
                    dl_dmean = np.sum(dl_dx * (- 1 / layer.std), axis = 0) + dl_dstd * (np.sum(-2 * layer.x * layer.std, axis = 0) / len(layer.x))
                    loss = dl_dx / layer.std + dl_dstd * 2 * (layer.x * layer.std) / len(layer.x) + dl_dmean / len(layer.x)

            elif type(layer).__name__ == 'RNN':
                changes_by = 0
                changes_V = 0
                changes_bh = 0
                changes_W = 0
                changes_U = 0

                if "batch_first=True":
                    loss = np.transpose(loss, axes=[1, 2, 0])
                else:
                    loss = np.transpose(loss, axes=[0, 2, 1])

                temp_loss = np.zeros((layer.T, layer.E, loss.shape[2]))

                for t in range(layer.T):
                    temp_ = 0
                    if (layer.last_input == 0):
                        temp_ = loss[t] * layer.derivative(layer.output_no_activation[t])

                    elif (t >= layer.T - layer.last_input):
                        temp_ = loss[t - (layer.T - layer.last_input)] * layer.derivative(layer.output_no_activation[t - (layer.T - layer.last_input)])

                    if not (isinstance(temp_, int)):
                        changes_by += temp_.mean(axis=1)
                        changes_V += (temp_ @ layer.hidden_activation[t].T) / loss.shape[2]
                        for _ in range(temp_.shape[1]):
                            temp_loss[t][:, _] += Parameter.calling[layer][0].T @ ((Parameter.calling[layer][2].T @ temp_[:, _]) * layer.derivative(layer.hidden_no_activation[t])[:, _])

                        temp__ = 0
                        for k in range(1, t + 1):
                            temp___ = 1
                            for j in range(k + 1, t + 1):
                                temp___ *= Parameter.calling[layer][1].T @ layer.derivative(layer.hidden_no_activation[j])
                            temp__ += temp___ * layer.derivative(layer.hidden_no_activation[k])

                            changes_bh += Parameter.calling[layer][2].T @ temp_ * temp__
                            changes_W += Parameter.calling[layer][2].T @ temp_ * temp__ @ layer.hidden_activation[k - 1].T
                            changes_U += Parameter.calling[layer][2].T @ temp_ * temp__ @ layer.input[k].T

                clip_ = 5
                changes_bh = np.clip(changes_bh.mean(axis=1), -clip_, clip_)
                changes_by = np.clip(changes_by, -clip_, clip_)
                changes_V = np.clip(changes_V, -clip_, clip_)
                changes_W = np.clip(changes_W, -clip_, clip_)
                changes_U = np.clip(changes_U, -clip_, clip_)
                self.backward_list.append([changes_U, changes_W, changes_V, changes_bh, changes_by])

                temp_loss = np.clip(temp_loss, -clip_, clip_)

                if "batch_first=True":
                    loss = np.transpose(temp_loss, axes=[2, 0, 1])
                else:
                    loss = np.transpose(temp_loss, axes=[0, 2, 1])

            elif type(layer).__name__ in ('Sigmoid', 'Relu', 'Leaky_relu', 'Tanh'):
                if type(Parameter.layers[::-1][index + 1]).__name__ == 'Conv':
                    loss = loss @ layer.derivative()
                else:
                    loss = loss * layer.derivative()
                self.backward_list.append([])

class Softmax():

    def __call__(self, z):
        if z.ndim == 1:
            return np.exp(z) / np.sum(np.exp(z))
        else:
            return np.exp(z) / np.sum(np.exp(z), axis=1).reshape(-1, 1)


