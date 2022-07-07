import os
import sys

import torch
import torch.utils.data as data

import numpy as np
import torchvision.transforms
from PIL import Image
def load_data_fashion_mnist(batch_size,resize = None,root = '~/Datasets/FashionMnist'):
    if sys.platform.startswith('win'):
        num_works = 0
    else:
        num_works = 4
    trans = []
    if resize:
        trans.append(torchvision.transforms.Resize(size = resize))
    trans.append(torchvision.transforms.ToTensor())

    transform = torchvision.transforms.Compose(trans)
    mnist_train = torchvision.datasets.CIFAR10(root = root,train=True,download=True,transform=transform)
    mnist_test = torchvision.datasets.CIFAR10(root = root,train=False,download=True,transform=transform)

    train_iter = torch.utils.data.DataLoader(mnist_train, batch_size = batch_size,shuffle=True,num_workers=num_works)
    test_iter = torch.utils.data.DataLoader(mnist_test, batch_size=batch_size, shuffle = True, num_workers=num_works)

    return train_iter, test_iter

