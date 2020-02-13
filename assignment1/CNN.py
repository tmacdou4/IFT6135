import torch
import torch.nn as nn
import numpy as np
import torch.optim as optim
from solution import *

import torchvision
import torchvision.transforms

from_numpy = torch.from_numpy

batch_size = 64
num_epochs = 10
store_every = 200
cuda = torch.cuda.is_available()
if cuda:
    print('cuda is available')
else:
    print('cuda is not available')

train, valid, test = load_mnist()

train = (train[0], np.argmax(train[1], axis=1))
valid = (valid[0], np.argmax(valid[1], axis=1))

train_dataset = torch.utils.data.TensorDataset(torch.Tensor(train[0]), torch.tensor(train[1], dtype=torch.int64))
valid_dataset = torch.utils.data.TensorDataset(torch.Tensor(valid[0]), torch.tensor(valid[1], dtype=torch.int64))
#test_dataset = torch.utils.data.TensorDataset(torch.Tensor(test[0]), torch.Tensor(test[1]))

train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
valid_loader = torch.utils.data.DataLoader(valid_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
#test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False, num_workers=2)


class Flatten(nn.Module):
    def forward(self, x):
        x = x.view(x.size(0), -1)
        return x

def accuracy(proba, y):
    correct = torch.eq(proba.max(1)[1], y).sum().type(torch.FloatTensor)
    return correct / y.size(0)


def evaluate(dataset_loader, criterion):
    LOSSES = 0
    COUNTER = 0
    for batch in dataset_loader:
        optimizer.zero_grad()

        x, y = batch

        x = x.view(-1, 1, 28, 28)
        y = y.view(-1)
        if cuda:
            x = x.cuda()
            y = y.cuda()

        loss = criterion(model(x), y)
        n = y.size(0)
        LOSSES += loss.sum().data.cpu().numpy() * n
        COUNTER += n
    return LOSSES / float(COUNTER)

def train(model, criterion, optimizer, num_epochs):
    LOSSES = 0
    COUNTER = 0
    ITERATIONS = 0
    learning_curve_nll_train = list()
    learning_curve_nll_valid = list()
    learning_curve_acc_train = list()
    learning_curve_acc_valid = list()
    for e in range(num_epochs):
        for batch in train_loader:
            optimizer.zero_grad()

            x, y = batch
            x = x.view(-1, 1, 28, 28)
            y = y.view(-1)
            if cuda:
                x = x.cuda()
                y = y.cuda()

            loss = criterion(model(x), y)
            loss.backward()
            optimizer.step()

            n = y.size(0)
            LOSSES += loss.sum().data.cpu().numpy() * n
            COUNTER += n
            ITERATIONS += 1
            if ITERATIONS % (200) == 0:
                avg_loss = LOSSES / float(COUNTER)
                LOSSES = 0
                COUNTER = 0
                print(" Iteration {}: TRAIN {}".format(
                    ITERATIONS, avg_loss))

        train_loss = evaluate(train_loader, criterion)
        learning_curve_nll_train.append(train_loss)
        valid_loss = evaluate(valid_loader, criterion)
        learning_curve_nll_valid.append(valid_loss)

        train_acc = evaluate(train_loader, accuracy)
        learning_curve_acc_train.append(train_acc)
        valid_acc = evaluate(valid_loader, accuracy)
        learning_curve_acc_valid.append(valid_acc)

        print(" [NLL] TRAIN {} / TEST {}".format(
            train_loss, valid_loss))
        print(" [ACC] TRAIN {} / TEST {}".format(
            train_acc, valid_acc))

    #return learning_curve_acc_valid[num_epochs-1], learning_curve_nll_valid[num_epochs-1]
    return learning_curve_acc_valid, learning_curve_nll_valid

def build_model(d):
    model = nn.Sequential(
        nn.Conv2d(1, 64, 5),  # (5x5x1+1)x32 parameters
        nn.Dropout2d(p=d),
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(64, 128, 5),  # (5x5x32+1)x64 parameters
        nn.ReLU(),
        nn.MaxPool2d(2, 2),
        nn.Conv2d(128, 225, 4),  # (4x4x64+1)x128 parameters
        nn.ReLU(),
        Flatten(),
        nn.Linear(225, 100),
        nn.ReLU(),
        nn.Linear(100, 10)
    )

    if cuda:
        model.cuda()

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=0.02, momentum=0.0)

    return model, criterion, optimizer


final_valid_accs = []
final_valid_losses = []
for d in [0]:
    model, criterion, optimizer = build_model(d)
    acc, loss = train(model, criterion, optimizer, 10)
    #final_valid_accs.append(acc)
    #final_valid_losses.append(loss)
    print(acc)
    print(loss)

print(final_valid_accs)
print(final_valid_losses)