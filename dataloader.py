import numpy as np

class DataBatcher():

    def __init__(self, data, batch_size=64, shuffle=True, flatten = True):

        self.data = data
        self.index = 0
        self.items = []
        self.flatten = flatten

        self.max = data.shape[0] // batch_size + (1 if data.shape[0] % batch_size != 0 else 0)

        if shuffle == True:
            self.data = np.random.permutation(self.data)

        for _ in range(self.max):
            self.items.append(self.data[batch_size * _: batch_size * (_ + 1)])

    def __iter__(self):
        return self


    # returns (features, targets)
    def __next__(self):
        if self.index < self.max:
            value = self.items[self.index]
            self.index += 1
            if self.flatten:
                return value[:, 1:], value[:, 0].reshape(-1, 1)
            else:
                return value[:, 1:].reshape(value.shape[0], 28, 28), value[:, 0].reshape(-1, 1)
        else:
            self.index = 0
            raise StopIteration
