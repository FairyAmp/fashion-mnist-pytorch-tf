#!/usr/bin/env python3

import numpy as np
import torch
from torch import nn
import torch.nn.functional as F

from torch.utils.data import DataLoader
from torchvision import datasets
import torchvision.transforms as transforms
from matplotlib import pyplot as plt

'''
using pytorch instead of tensorflow
tensorflow also has the
FashionMNIST dataset in its library
but most examples I saw were using torch
seems easier to use so trying this out
also tried to write this using tf
but it seems pretty bugged
so submitting the torch version instead
'''

device = (
    "cpu"
)

class NeuralNetwork(nn.Module):
    def __init__(self):
        super().__init__()
        self.conv1 = nn.Conv2d(1, 28, 3) # 26 x 26 x 28
        self.relu = nn.ReLU(inplace=True)
        self.pool = nn.MaxPool2d(2) # 13 x 13 x 28
        self.conv2 = nn.Conv2d(28, 56, 3) # 11 x 11 x 56
        self.flatten = nn.Flatten()
        self.linear1 = nn.Linear(11*11*56, 56)
        self.linear2 = nn.Linear(56, 10)

    def forward(self, x):
        x = self.conv1(x)
        x = self.relu(x)
        x = self.pool(x)
        x = self.conv2(x)
        x = self.relu(x)
        x = self.flatten(x)
        x = self.linear1(x)
        x = self.relu(x)
        x = self.linear2(x)
        return F.log_softmax(x, dim=1)

def run(model, loader, optimizer, mode = 'train'):
    if mode == 'train':
        model.train()
    else:
        model.eval()

    tot = len(loader.dataset)
    tot_loss = 0
    tot_cnt = 0
    tot_correct = 0
    for batch, (X, y) in enumerate(loader):
        X, y = X.to(device), y.to(device)

        # Compute prediction error
        output = model(X)
        loss = F.nll_loss(output, y)

        if mode == 'train':
            # Backpropagation
            loss.backward()
            optimizer.step()
            optimizer.zero_grad()

        tot_loss += loss*len(X)
        tot_cnt += len(X)
        pred = output.argmax(dim=1, keepdim=True)
        correct = pred.eq(y.view_as(pred)).sum().item()
        tot_correct += correct
        accuracy = 100. * correct / len(X)
        if mode == 'train' and batch % 100 == 99:
            print(f"[{tot_cnt:>5d}/{len(loader.dataset):>5d}] loss: {loss.item():>5f}, accuracy: {accuracy:>2.4f}%")
    accuracy = 100. * tot_correct / tot
    loss = tot_loss / tot
    print(f'{mode.upper():>6s}: Average loss: {loss:>5f}, Accuracy: {tot_correct:>5d}/{tot:>5d} ({accuracy:>2.4f}%)')
    return accuracy

def main():
    lr = 0.001
    epochs = 10
    batch_size = 32
    torch.manual_seed(7)
    # Download training data from open datasets.
    # Data is already within [0,1]
    transform = transforms.Compose(
        [transforms.ToTensor()])
    train_set = datasets.FashionMNIST('./data', train=True, transform=transform, download=True)
    test_set = datasets.FashionMNIST('./data', train=False, transform=transform, download=True)
    # Use last 12000 as validation set
    valid_set = torch.utils.data.Subset(train_set, range(48000,60000))
    train_set = torch.utils.data.Subset(train_set, range(48000))
    # Class labels
    labels = ('T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
            'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle Boot')

    # Create data loaders
    train_loader = DataLoader(train_set, batch_size=batch_size, shuffle=True)
    valid_loader = DataLoader(valid_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    sampler = DataLoader(test_set)

    # Initialize model, optimizer
    model = NeuralNetwork().to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    total_params = sum(p.numel() for p in model.parameters())
    print(f"Number of trainable parameters: {total_params}")

    train_accuracies = []
    valid_accuracies = []
    for epoch in range(1, epochs + 1):
        print(f'epoch {epoch} / {epochs}:')
        train_acc = run(model, train_loader, optimizer, 'train')
        valid_acc = run(model, valid_loader, None, 'valid')
        train_accuracies.append(train_acc)
        valid_accuracies.append(valid_acc)

    x = range(1, epochs + 1)
    plt.figure(1)
    plt.plot(x, train_accuracies, 's-')
    plt.plot(x, valid_accuracies, 'o--')
    plt.legend(['train','validation'])
    plt.xlabel('Epoch')
    plt.ylabel('Accuracy(%)')
    plt.grid()
    plt.savefig('accuracies.png')
    # Evaluate the accuracy on test set
    test_accuracy = run(model, test_loader, None, 'test')
    print(f'Accuracy on Test set is {test_accuracy:>2.4f}%')

    # Examples of misclassification
    sampler = DataLoader(test_set)
    misclass = [None]*10
    found = [False]*10
    cnt = 0
    for X,y in sampler:
        output = model(X)
        pred = output.argmax(dim=1, keepdim=True)
        pred, y = pred.item(), y.item()
        if pred != y and not found[y]:
            found[y] = True
            cnt += 1
            misclass[y] = (pred, X)
            if cnt == 10: break
            
    plt.figure(figsize=(12, 6))

    for y, (pred, X) in enumerate(misclass):
        plt.subplot(2, 5, y + 1)
        plt.imshow(X.cpu().detach().numpy().reshape((28, 28, 1)))
        plt.title(f'Truth:{labels[y]:>5s}\nPredicted:{labels[pred]:>5s}')

    plt.tight_layout()
    plt.savefig('examples.png')

if __name__ =="__main__":
    main()

# Observations
'''
     from the accuracies graph, it seems like there might be some overfitting
     the accuracy of the train set gets closer to 100% with each epoch
     which is what I expect, but the accuracy of the validation set
     seems to even out at around 92% after the third epoch
     Accuracy on test set is around 90.5%
'''
