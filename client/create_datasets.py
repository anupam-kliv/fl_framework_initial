

from random import randint, shuffle
import subprocess
import os
import shutil
from tqdm import tqdm

import torch
from torchvision.datasets import MNIST
from torchvision.transforms import transforms
from torch.utils.data import random_split

mnist_dataset = MNIST("./", train=True, download=True, transform=transforms.ToTensor())

#number of datasets to create
num_of_clients = 3

client_sizes = []
remaining = len(mnist_dataset)
for _ in range(num_of_clients):
    try:
        client_sizes.append( randint(1, remaining + 1) )
        remaining -= client_sizes[-1]
    except ValueError:
        client_sizes.append(0)
client_sizes[-1] += remaining

datasets = random_split(mnist_dataset, client_sizes)
torch.save(datasets, "client_datasets.pt")