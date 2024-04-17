import numpy as np
from pycandle.metrics import accuracy

def train(model, train_batches, test_batches, loss, optimizer, n_epoch=6):
    for epoch in range(n_epoch):

        for i, batch in enumerate(train_batches):

            X_batch, y_batch = batch

            logits = model(X_batch)

            loss(logits, y_batch)

            loss.backward()

            optimizer.step(loss)

            if i % 50 == 0:
                print(epoch, i, evaluate(model, test_batches))

    return model


def evaluate(model, dataloader):
    y_pred_list = []
    y_true_list = []
    for i, batch in enumerate(dataloader):
        X_batch, y_batch = batch
        logits = model(X_batch)
        y_pred_list.extend(logits)
        y_true_list.extend(np.int_(np.arange(0, 10) == y_batch))

    return accuracy(np.array(y_true_list), np.array(y_pred_list))