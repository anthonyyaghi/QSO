#!/usr/bin/env python
# coding: utf-8

from __future__ import print_function
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
from torchvision import datasets, transforms
import torchvision.models as models
from torch.optim.lr_scheduler import StepLR
import matplotlib.pyplot as plt
from scipy.stats import norm
import time
import math
from torch.optim.optimizer import Optimizer, required


class EmaFilter:
    def __init__(self, sf):
        self.prev = None
        self.ema_weight = sf

    def filter(self, t_val: torch.Tensor):
        if self.prev is None:
            self.prev = t_val
            return self.prev
        else:
            self.prev = t_val * self.ema_weight + self.prev * (1 - self.ema_weight)
            return self.prev


class XGen:
    def __init__(self, sf):
        self.ema_filter = EmaFilter(sf)

    def compute(self, gz, pgz, xdata, pxdata):
        x = self.ema_filter.filter((gz - pgz) / (xdata - pxdata))
        x = x.abs()
        x[torch.isnan(x)] = 0
        return torch.clamp(x, 0.01, 1)


class QSOA(Optimizer):
    r"""Implements Quasi Second Order Optimization

    Args:
        params (iterable): iterable of parameters to optimize or dicts defining
            parameter groups
        lr (float): learning rate

    Example:
        >>> optimizer = QSOA(model.parameters(), lr=0.01)
        >>> optimizer.zero_grad()
        >>> loss_fn(model(input), target).backward()
        >>> optimizer.step()

    """

    def __init__(self, params, lr=required):
        if lr is not required and lr < 0.0:
            raise ValueError("Invalid learning rate: {}".format(lr))
        defaults = dict(lr=lr)
        super(QSOA, self).__init__(params, defaults)

    def __setstate__(self, state):
        super(QSOA, self).__setstate__(state)

    def step(self, closure=None):
        """Performs a single optimization step.

        Arguments:
            closure (callable, optional): A closure that reevaluates the model
                and returns the loss.
        """
        loss = None
        if closure is not None:
            loss = closure()

        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                grad = p.grad.data
                xdata = p.data

                if hasattr(p, 'pxdata'):
                    pgrad = getattr(p, 'pgrad')
                    pxdata = getattr(p, 'pxdata')
                    xk = getattr(p, 'xk')
                    dg = xk.compute(grad, pgrad, xdata, pxdata)
                    setattr(p, 'pxdata', xdata.clone())
                    p.data.add_(-group['lr'], grad * dg)
                    setattr(p, 'pgrad', grad.clone())
                else:
                    setattr(p, 'pxdata', xdata.clone())
                    p.data.add_(-group['lr'], grad)
                    setattr(p, 'pgrad', grad.clone())
                    setattr(p, 'xk', XGen(0.5))

        return loss


batch_size = 32
transform = transforms.Compose(
    [transforms.ToTensor(),
     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

train_set = torchvision.datasets.CIFAR10(root='./data', train=True,
                                         download=True, transform=transform)
train_loader = torch.utils.data.DataLoader(train_set, batch_size=batch_size,
                                           shuffle=True, num_workers=2)

test_set = torchvision.datasets.CIFAR10(root='./data', train=False,
                                        download=True, transform=transform)
test_loader = torch.utils.data.DataLoader(test_set, batch_size=batch_size,
                                          shuffle=False, num_workers=2)

classes = ('plane', 'car', 'bird', 'cat',
           'deer', 'dog', 'frog', 'horse', 'ship', 'truck')


def train(m_epoch):
    net.train()
    for batch_idx, (data, target) in enumerate(train_loader):
        data, target = data.to(device), target.to(device)
        optimizer.zero_grad()
        output = net(data)
        loss = F.cross_entropy(output, target)
        loss.backward()
        optimizer.step()
        if batch_idx % log_interval == 0:
            print('Train Epoch: {} [{}/{} ({:.0f}%)]\tLoss: {:.6f}'.format(
                m_epoch, batch_idx * len(data), len(train_loader.dataset),
                         100. * batch_idx / len(train_loader), loss.item()))
            train_losses.append(loss.item())
            train_counter.append((batch_idx * 4) + ((epoch - 1) * len(train_loader.dataset)))


def test():
    net.eval()
    test_loss = 0
    correct = 0
    with torch.no_grad():
        for data, target in test_loader:
            data, target = data.to(device), target.to(device)
            output = net(data)
            test_loss += F.cross_entropy(output, target, reduction='sum').item()
            pred = output.data.max(1, keepdim=True)[1]
            correct += pred.eq(target.data.view_as(pred)).sum()
    test_loss /= len(test_loader.dataset)
    test_losses.append(test_loss)
    test_acc.append(100. * correct / len(test_loader.dataset))
    print('\nTest set: Avg. loss: {:.4f}, Accuracy: {}/{} ({:.0f}%)\n'.format(
        test_loss, correct, len(test_loader.dataset),
        100. * correct / len(test_loader.dataset)))


n_epochs = 10
save_epochs = 5

log_interval = math.ceil(50000/batch_size/35)
# random_seed = torch.randint(0, 100, (1, 1)).item()
random_seed = 16
torch.backends.cudnn.enabled = True
torch.manual_seed(random_seed)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)

# SGD

start_time = time.time()

net = models.resnet18()
net.fc = nn.Linear(512, 10)
net = net.to(device)

optimizer = optim.SGD(net.parameters(), lr=0.001)
train_losses = []
train_counter = []
test_losses = []
test_acc = []

# load ?
# net.load_state_dict(torch.load('nets/sgd_10'))
# test_losses = torch.load('nets/test_losses_sgd_10')
# test_acc = torch.load('nets/test_acc_sgd_10')
# train_losses = torch.load('nets/train_losses_sgd_10')
# train_counter = torch.load('nets/train_counter_sgd_10')
start_epoch = 1

for epoch in range(start_epoch, n_epochs + 1):
    train(epoch)
    test()
    if epoch % save_epochs == 0:
        torch.save(net.state_dict(), f'nets/sgd_{epoch}')
        torch.save(test_losses, f'nets/test_losses_sgd_{epoch}')
        torch.save(test_acc, f'nets/test_acc_sgd_{epoch}')
        torch.save(train_losses, f'nets/train_losses_sgd_{epoch}')
        torch.save(train_counter, f'nets/train_counter_sgd_{epoch}')

sgd_test_losses = test_losses

mean_sgd, std_sgd = norm.fit(sgd_test_losses)
print(f'SGD mean: {mean_sgd}\tSGD std: {std_sgd}')
print("--- %s seconds ---" % (time.time() - start_time))

# QSO

start_time = time.time()

net = models.resnet18()
net.fc = nn.Linear(512, 10)
net = net.to(device)

optimizer = QSOA(net.parameters(), lr=0.001)
train_losses = []
train_counter = []
test_losses = []
test_acc = []

# load ?
# net.load_state_dict(torch.load('nets/qso_10'))
# test_losses = torch.load('nets/test_losses_qso_10')
# test_acc = torch.load('nets/test_acc_qso_10')
# train_losses = torch.load('nets/train_losses_qso_10')
# train_counter = torch.load('nets/train_counter_qso_10')
start_epoch = 1

for epoch in range(start_epoch, n_epochs + 1):
    train(epoch)
    test()
    if epoch % save_epochs == 0:
        torch.save(net.state_dict(), f'nets/qso_{epoch}')
        torch.save(test_losses, f'nets/test_losses_qso_{epoch}')
        torch.save(test_acc, f'nets/test_acc_qso_{epoch}')
        torch.save(train_losses, f'nets/train_losses_qso_{epoch}')
        torch.save(train_counter, f'nets/train_counter_qso_{epoch}')

qso_test_losses = test_losses

mean_qso, std_qso = norm.fit(qso_test_losses)
print(f'QSO mean: {mean_qso}\tQSO std: {std_qso}')
print("--- %s seconds ---" % (time.time() - start_time))

# Load test losses
sgd_test_losses = torch.load('nets/test_losses_sgd_10')
qso_test_losses = torch.load('nets/test_losses_qso_10')

# Load train losses
sgd_train_losses = torch.load('nets/train_losses_sgd_10')
qso_train_losses = torch.load('nets/train_losses_qso_10')

# Load train counter
sgd_train_counter = torch.load('nets/train_counter_sgd_10')
qso_train_counter = torch.load('nets/train_counter_qso_10')

# Load test errors
sgd_test_acc = torch.load('nets/test_acc_sgd_10')
qso_test_acc = torch.load('nets/test_acc_qso_10')

current_epoch = 10

test_counter = [i * len(train_loader.dataset) for i in range(1, current_epoch + 1)]
train_counter = []
fig = plt.figure()
plt.plot(test_counter, sgd_test_losses, color='red')
plt.plot(sgd_train_counter, sgd_train_losses, color='blue')
plt.legend(['Test Loss', 'Train Loss'], loc='upper right')
fig.savefig('sgd_losses.png')

fig = plt.figure()
plt.plot(test_counter, qso_test_losses, color='red')
plt.plot(sgd_train_counter, qso_train_losses, color='blue')
plt.legend(['Test Loss', 'Train Loss'], loc='upper right')
fig.savefig('qso_losses.png')

fig = plt.figure()
plt.plot(test_counter, sgd_test_acc, color='red')
plt.plot(test_counter, qso_test_acc, color='blue')
plt.legend(['SGD', 'QSO'], loc='upper right')
fig.savefig('test_acc.png')
