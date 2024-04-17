import pycandle.supplementary as supplementary
import pycandle.dataloader as dataloader
import pycandle.nn as nn
import pycandle.optim as optim
import pycandle. functional as F


'''
examples of models build with pycandle
'''

# digit image dataset (28x28)
# classic neural network
class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 200)
        self.act1 = F.Relu()
        self.linear2 = nn.Linear(200, 50)
        self.act2 = F.Tanh()
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

# digit image dataset (28x28)
# classic neural network with batchnorm

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 200)
        self.batch1 = nn.BatchNorm(200)
        self.act1 = F.Relu()
        self.linear2 = nn.Linear(200, 50)
        self.batch2 = nn.BatchNorm(50)
        self.act2 = F.Relu()
        self.linear3 = nn.Linear(50, 10)
        self.sftmx = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.batch2(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.sftmx(x)
        return x

# digit image dataset (28x28)
# classic neural network with dropout

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.linear1 = nn.Linear(784, 200)
        self.drop1 = nn.DropOut(q = 0.2)
        self.act1 = F.Relu()
        self.linear2 = nn.Linear(200, 50)
        self.drop1 = nn.DropOut(q = 0.5)
        self.act2 = F.Relu()
        self.linear3 = nn.Linear(50, 10)
        self.sftmx = nn.Softmax()

    def forward(self, x):
        x = self.linear1(x)
        x = self.drop1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.drop1(x)
        x = self.act2(x)
        x = self.linear3(x)
        x = self.sftmx(x)
        return x


# digit image dataset (28x28)
# classic convolutional neural network and batchborm

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv(input_channels=1, output_channels=3, kernel_size=(3, 3))  # 26x26
        self.batch0 = nn.BatchNorm(3)
        self.act1 = F.Tanh()
        self.conv2 = nn.Conv(input_channels=3, output_channels=5, kernel_size=(4, 4))  # 23x23
        self.batch1 = nn.BatchNorm(5)
        self.act2 = F.Tanh()
        self.conv3 = nn.Conv(input_channels=5, output_channels=5, kernel_size=(4, 4))  # 20x20
        self.batch2 = nn.BatchNorm(5)
        self.act3 = F.Tanh()
        self.flatten = nn.Flatten()  # 20 * 20 * 5
        self.fc1 = nn.Linear(2000, 500)
        self.act4 = F.Tanh()
        self.fc2 = nn.Linear(500, 50)
        self.act5 = F.Tanh()
        self.fc3 = nn.Linear(50, 10)
        self.sftmx = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch0(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.batch1(x)
        x = self.act2(x)
        x = self.conv3(x)
        x = self.batch2(x)
        x = self.act3(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act4(x)
        x = self.fc2(x)
        x = self.act4(x)
        x = self.fc3(x)
        x = self.sftmx(x)
        return x

# digit image dataset (28x28)
# convolutional neural network with pooling layer

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv(input_channels=1, output_channels=3, kernel_size=(3, 3))  # 26x26
        self.act1 = F.Relu()
        self.pool1 = nn.MaxPool(kernel_size=(2, 2))  # 13x13
        self.conv2 = nn.Conv(input_channels=3, output_channels=5, kernel_size=(4, 4))  # 10x10
        self.act2 = F.Relu()
        self.flatten = nn.Flatten()  # 500
        self.fc1 = nn.Linear(500, 100)
        self.act3 = F.Relu()
        self.fc2 = nn.Linear(100, 10)
        self.sftmx = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.act3(x)
        x = self.fc2(x)
        x = self.sftmx(x)
        return x


# digit image dataset (28x28)
# classic convolutional neural network with pooling and batchnorm

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv(input_channels=1, output_channels=3, kernel_size=(3, 3))  # 26x26
        self.batch1 = nn.BatchNorm(3)
        self.act1 = F.Relu()
        self.pool1 = nn.MaxPool(kernel_size=(2, 2))  # 13x13
        self.conv2 = nn.Conv(input_channels=3, output_channels=5, kernel_size=(4, 4))  # 10x10
        self.batch2 = nn.BatchNorm(5)
        self.act2 = F.Relu()
        self.flatten = nn.Flatten()  # 500
        self.fc1 = nn.Linear(500, 10)

        self.sftmx = nn.Softmax()


    def forward(self, x):
        x = self.conv1(x)
        x = self.batch1(x)
        x = self.act1(x)
        x = self.pool1(x)
        x = self.conv2(x)
        x = self.batch2(x)
        x = self.act2(x)
        x = self.flatten(x)
        x = self.fc1(x)
        x = self.sftmx(x)
        return x


# digit image dataset (28x28)
# recurrent neural network

class SimpleNet(nn.Module):
    def __init__(self):
        super().__init__()
        self.rnn1 = nn.RNN(28, 15, 10)
        self.rnn2 = nn.RNN(10, 5, 5, last_input = 10)
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(input_channels= 10 * 5, output_channels=50)
        self.act1 = F.Relu()
        self.linear2 = nn.Linear(input_channels=50, output_channels=10)

        self.sftmx = nn.Softmax()


    def forward(self, x):
        x = self.rnn1(x)
        x = self.rnn2(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.act1(x)
        x = self.linear2(x)
        x = self.sftmx(x)
        return x