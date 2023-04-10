import os
import torch
import torch.optim as optim
from tqdm import tqdm
import time
import numpy as np
from torchvision import transforms,datasets
import torch.optim as optim
import torchvision

#from task_5.server.get_config import get_config

#config = get_config()


def get_data(config):
    #print(config)
    if config['dataset'] == 'MNIST':
        apply_transform = transforms.Compose([transforms.Resize(config["resize_size"]), transforms.ToTensor()])
        trainset = datasets.MNIST(root='./MNIST', train=True, download=True, transform=apply_transform)
        testset = datasets.MNIST(root='./MNIST', train=False, download=True, transform=apply_transform)
    if config['dataset'] == 'FashionMNIST':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.FashionMNIST(root='./FashionMNIST', train=True, download=True, transform=apply_transform)
        testset = datasets.FashionMNIST(root='./FashionMNIST', train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR10':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR10(root='./CIFAR10', train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR10(root='./CIFAR10', train=False, download=True, transform=apply_transform)

    if config['dataset'] == 'CIFAR100':
        apply_transform = transforms.Compose([transforms.Resize(config['resize_size']), transforms.ToTensor()])
        trainset = datasets.CIFAR100(root='./CIFAR100', train=True, download=True, transform=apply_transform)
        testset = datasets.CIFAR100(root='./CIFAR100', train=False, download=True, transform=apply_transform)

    return trainset, testset
