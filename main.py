
import sys
path = ... # add your path to the pycandle lib
sys.path.append(path)

import pycandle.supplementary as supplementary
import pycandle.dataloader as dataloader
import pycandle.nn as nn
import pycandle.optim as optim
import pycandle.functional as F

import pandas as pd
import numpy as np

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 200)
        self.act1 = F.Relu()
        self.linear2 = nn.Linear(200, 50)
        self.act2 = F.Relu()
        self.linear3 = nn.Linear(50, 10)
        self.sftmx = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.sftmx(x)
        return x


if __name__ == '__main__':

    data = pd.read_csv('train.csv')
    data = np.array(data)
    data = data.astype(float)
    np.random.shuffle(data)
    data[:, 1:] = data[:, 1:] / 255
    test_data = data[0:5000]
    train_data = data[5000:42000]
    test_batches = dataloader.DataBatcher(test_data, 64, True, flatten = True)
    train_batches = dataloader.DataBatcher(train_data, 64, True, flatten = True)

    model = SimpleNet()

    loss_fn = nn.CrossEntropyLoss(l2_reg=0.)

    optimizer = optim.ADAM(model, learning_rate=1e-3, momentum=0.5, ro = 0.5)
    model = supplementary.train(model, train_batches, test_batches, loss_fn, optimizer, 2)

    optimizer = optim.ADAM(model, learning_rate=5e-4, momentum=0.5, ro=0.1)
    model = supplementary.train(model, train_batches, test_batches, loss_fn, optimizer, 3)
