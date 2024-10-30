import numpy
from collections import defaultdict
try:
    import cupy
except:
    pass


'''
Tensor class is a full analog of pytorch.tensor
New class overloads most popular numpy functions ( +, @, ones, reshape etc.)
So new class also provides all gradient computations with computational graph.
You can also run all computations on gpu (if available) by setting .to('gpu')
All gpu computations provided thanks to cupy library.
a big thanks to this repo: https://github.com/sradc/SmallPebble/blob/main/smallpebble/smallpebble.py#L708 , most of the cool ideas are taken from there
'''

np = numpy

class Tensor:

    device = 'cpu'

    def __init__(self, value, requires_grad=True, local_gradients=()):
        self.value = np.array(value)
        self.local_gradients = local_gradients
        self.shape = self.value.shape
        self.requires_grad = requires_grad
        self.ndim = self.value.ndim

    def to(self, device = 'cpu'):
        global np
        if device == 'cpu':
            Tensor.device = 'cpu'
            np = numpy
        elif device == 'gpu':
            Tensor.device = 'gpu'
            np = cupy
        else:
            raise Exception("No device has found")

    def __add__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value + other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('add', self, lambda x: x))

        if other.requires_grad == True:
            temp.append(('add', other, lambda x: x))

        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __radd__(self, other):

        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value + other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('radd', self, lambda x: x))
        if other.requires_grad == True:
            temp.append(('radd', other, lambda x: x))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __sub__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value - other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('sub', self, lambda x: x))
        if other.requires_grad == True:
            temp.append(('sub', other, lambda x: -x))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __rsub__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = other.value - self.value
        temp = []
        if self.requires_grad == True:
            temp.append(('rsub', self, lambda x: -x))
        if other.requires_grad == True:
            temp.append(('rsub', other, lambda x: x))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __mul__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value * other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('mul', self, lambda x: x * other.value))
        if other.requires_grad == True:
            temp.append(('mul', other, lambda x: x * self.value))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __rmul__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value * other.value
        temp = []
        if self.requires_grad == True:
            temp.append(('rmul', self, lambda x: x * other.value))
        if other.requires_grad == True:
            temp.append(('rmul', other, lambda x: x * self.value))
        local_gradients = tuple(temp)
        return Tensor(value, local_gradients=local_gradients)

    def __matmul__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = self.value @ other.value
        local_gradients = (('matmul', self, lambda x: x @ other.value.T), ('matmul', other, lambda x: self.value.T @ x))
        return Tensor(value, local_gradients=local_gradients)

    def __rmatmul__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other, requires_grad=False)
        value = other.value @ self.value
        local_gradients = (('rmatmul', other, lambda x: other.value.T @ x), ('rmatmul', self, lambda x: x @ self.value.T))
        return Tensor(value, local_gradients=local_gradients)

    @staticmethod
    def inv(a):
        if a.requires_grad == True:
            value = 1. / a.value
            local_gradients = (('inv', a, lambda x: x * -1. / (a.value ** 2)),)
        else:
            value = 1. / a.value
            local_gradients = ((),)

        return Tensor(value, local_gradients=local_gradients)

    def __truediv__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return Tensor.__mul__(self, Tensor.inv(other))

    def __rtruediv__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return Tensor.__rmul__(Tensor.inv(self), other)

    def __neg__(self):
        return Tensor.__mul__(self, -1)

    def __pow__(self, n):
        value = self.value ** n
        local_gradients = (('pow', self, lambda x: x * np.ones(self.shape) * n * (self.value ** (n - 1))),)
        return Tensor(value, local_gradients=local_gradients)

    def __eq__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value == other.value

    def __lt__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value < other.value

    def __le__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value <= other.value

    def __gt__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value > other.value

    def __ge__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value >= other.value

    def __ne__(self, other):
        if other.__class__.__name__ != 'Tensor':
            other = Tensor(other)
        return self.value != other.value

    def __getitem__(self, item):
        temp = np.zeros(self.shape)
        temp[item] = 1

        def multiply_by_locgrad(path_value):
            _ = np.zeros(self.shape)
            _[item] = path_value
            return _

        local_gradients = (('getitem', self, multiply_by_locgrad),)
        return Tensor(self.value[item], local_gradients=local_gradients)

    def __setitem__(self, key, val):
        self.value[key] = val

    def detach(self):
        return Tensor(self.value)

    @staticmethod
    def sin(a):
        value = np.sin(a.value)
        local_gradients = (
            ('sin', a, lambda x: x * np.cos(a.value)),
        )
        return Tensor(value, local_gradients=local_gradients)

    @staticmethod
    def cos(a):
        value = np.cos(a.value)
        local_gradients = (
            ('cos', a, lambda x: x * -np.sin(a.value)),
        )
        return Tensor(value, local_gradients=local_gradients)

    @staticmethod
    def exp(a):
        value = np.exp(a.value)
        local_gradients = (
            ('exp', a, lambda x: x * value),
        )
        return Tensor(value, local_gradients=local_gradients)

    @staticmethod
    def log(a):
        value = np.log(a.value)
        local_gradients = (
            ('log', a, lambda x: x * 1. / a.value),
        )
        return Tensor(value, local_gradients=local_gradients)

    @staticmethod
    def zeros(shape):
        return Tensor(np.zeros(shape))

    @staticmethod
    def sum(array, axis=None):
        local_gradients = (('sum', array, lambda x: x * np.ones(array.shape)),)
        return Tensor(np.sum(array.value, axis=axis), local_gradients=local_gradients)

    def reshape(self, *args):
        local_gradients = (('reshape', self, lambda x: x.reshape(self.shape)),)
        return Tensor(self.value.reshape(*args), local_gradients=local_gradients)

    @staticmethod
    def softmax(z):
        if z.ndim == 1:
            return Tensor.exp(z) / Tensor.sum(Tensor.exp(z))
        else:
            return Tensor.exp(z) / Tensor.sum(Tensor.exp(z), axis=1).reshape(-1, 1)

    @staticmethod
    def sliding_window_view(matrix, kernel_z, kernel_y, kernel_x):

        result = np.lib.stride_tricks.sliding_window_view(matrix.value, (1, kernel_z, kernel_y, kernel_x)).copy()

        def multiply_by_locgrad(path_value):  # TODO: a faster method
            temp = np.zeros(matrix.shape)
            if np.__name__ == 'numpy':
                np.add.at(np.lib.stride_tricks.sliding_window_view(temp, (1, kernel_z, kernel_y, kernel_x), writeable=True), None, path_value)
            elif np.__name__ == 'cupy':
                np.add.at(np.lib.stride_tricks.sliding_window_view(temp, (1, kernel_z, kernel_y, kernel_x)), None, path_value)

            return temp

        local_gradients = (('slide', matrix, multiply_by_locgrad),)
        return Tensor(result, local_gradients=local_gradients)

    @staticmethod
    def ones(shape):
        return Tensor(np.ones(shape))

    @staticmethod
    def sign(a):
        value = np.sign(a.value)
        return Tensor(value)

    def __hash__(self):
        return id(self)

    def __repr__(self):
        return np.array_repr(self.value)

    @staticmethod
    def sqrt(a):
        return Tensor.__pow__(a, 1 / 2)

    @staticmethod
    def mean(array, axis=None):
        if axis == None:
            local_gradients = (('mean', array, lambda x: x * np.ones(array.shape) / np.size(array.value)),)
            return Tensor(np.sum(array.value, axis=axis) / np.size(array.value), local_gradients=local_gradients)
        else:
            delimeter = 1
            if not isinstance(axis, int):
                for ax in axis:
                    delimeter = delimeter * array.shape[ax]
            else:
                delimeter = array.shape[axis]
            local_gradients = (('mean', array, lambda x: x * np.ones(array.shape) / delimeter),)
            return Tensor(np.sum(array.value, axis=axis) / delimeter, local_gradients=local_gradients)

    @staticmethod
    def std(array, axis=None):
        if axis == None:
            mean = Tensor.mean(array, axis=None)
            sqrt_sub = (array - mean) ** 2
            sum_ = Tensor.sum(sqrt_sub, axis=None) / np.size(array.value)
            return Tensor.sqrt(sum_)
        else:
            delimeter = 1
            if not isinstance(axis, int):
                for ax in axis:
                    delimeter = delimeter * array.shape[ax]
            else:
                delimeter = array.shape[axis]
            mean = Tensor.mean(array, axis=axis)
            if axis == 0:
                pass
            elif axis == 1:
                mean = mean.reshape((mean.shape[0], 1, *mean.shape[1:]))
            elif axis == 2:
                mean = mean.reshape((*mean.shape[:2], 1, *mean.shape[2:]))
            elif axis == 3:
                mean = mean.reshape((mean.shape[0], 1, 1, 1))
            elif isinstance(axis, (tuple, list)):
                temp = []
                for i in range(len(array.shape)):
                    if i in axis:
                        temp.append(1)
                    else:
                        temp.append(array.shape[i])
                mean = mean.reshape(tuple(temp))

            sqrt_sub = (array - mean) ** 2
            sum_ = Tensor.sum(sqrt_sub, axis=axis) / delimeter
            return Tensor.sqrt(sum_)

    def backward(self, loss=1):
        gradients = defaultdict(lambda: 0)

        def compute_gradients(variable, path_value):
            for oper_type, child, child_gradient_func in variable.local_gradients:
                value_path_to_child = child_gradient_func(path_value)
                gradients[child] += value_path_to_child
                compute_gradients(child, value_path_to_child)

        if isinstance(loss, Tensor):
            compute_gradients(self, path_value=loss.value)
        else:
            compute_gradients(self, path_value=np.array(loss))
        return gradients

    @staticmethod
    def relu(x):
        return x * (1 + Tensor.sign(x)) / 2

    @staticmethod
    def leaky_relu(x):
        return x * ((1 + Tensor.sign(x)) / 2 + 0.2 * (1 + Tensor.sign(-x)) / 2)

    @staticmethod
    def tanh(x):
        return (Tensor.exp(2 * x) - 1) / (Tensor.exp(2 * x) + 1)

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + Tensor.exp(-x))
