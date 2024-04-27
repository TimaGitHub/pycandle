# PyCandle 

![pycandle](https://github.com/TimaGitHub/Convolutional-Neural-Network-from-scratch/assets/70072941/0655a53c-79f2-4c9d-a18a-6a82ffb17cf4)


---

PyCandle is a deep learning library written completely from scratch in Python using numpy library.

PyCandle api  mimics  PyTorch api and gives a wide opportunity for creating models.

# Usage 
``` python
import pycandle.nn as nn

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

test_batches = dataloader.DataBatcher(test, 64, True, flatten = True)
train_batches = dataloader.DataBatcher(train, 64, True, flatten = True)

model = SimpleNet()

loss_fn = nn.CrossEntropyLoss(l2_reg=0.)

optimizer = optim.ADAM(model, learning_rate=1e-3, momentum=0.5, ro = 0.5)

model = supplementary.train(model, train_batches, test_batches, loss_fn, optimizer, n_epoch = 2)

```

## Highlights
PyCandle 
- contains most popular types of neural networks (Linear, Convolutional, Flatten, Min\MaxPool, BatchNorm, DropOut, RNN)
- contains most popular types of activation functions (Sigmoid, Relu, Leaky Relu, Tanh, Softmax)
- contains most popular types of gradient descent algorithms (Stochastic, Nesterov Accelerated Gradient, RMSProp, ADAM)
- gives the oportunity to create batches
- gives the oportunity to precisely tune Cross Entropy Loss with L2, L1 regularization parameters
- gives the oportunity to learn what's under the hood of deep learning algorithms (especcialy at back propagation)
- provides access to pytorch api without the need for users to adapt to a new library
- provided the ability to compute gradients with computational graph, see pycandle.Tensor()

### Examples
open examples.py to look at various types of NN
``` python
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

```

## To-Do List
- [x] provide all computations  using a computational graph and mimic pytorch.tensor() (done, see tensor.py, providing computational graph for all layers is left
- [ ] add LSTM, GRU, Transformers, GAN
- [x] add switching between NumPy and CuPy (numpy on gpu)
