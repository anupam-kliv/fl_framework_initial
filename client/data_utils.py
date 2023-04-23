import os
import torch
import random
import numpy as np

from PIL import Image
from torch.utils import data
from torchvision import transforms
import pickle
class customDataset(data.Dataset):

    def __init__(
        self,
        config,
        trainset,
        data_path,
        clientID = 0,
        aug = False,
        
    ):
        
        self.aug = aug
        #self.is_transform = is_transform
        self.config = config
        self.img_size = config["resize_size"]
        self.trainset = trainset
        #self.path = '../data/mnist/processed/'
        self.path1=data_path
        self.niid_degree = config["niid"]
        self.clientID = clientID
        #self.split = split
        self.mean = 33.3184
        self.stdv = 78.5675
        
        '''if self.split == 'fed' or self.split == 'train':
            self.file = torch.load(self.path+'training.pt')
        elif self.split == 'test':
            self.file = torch.load(self.path+'test.pt')'''
            
            
        #if self.split == 'fed':
        #self.data_idxs = torch.load('../data/mnist/data_distribution/data_split_niid_'+str(self.niid_degree)+'.pt')['datapoints'][clientID]
        self.data_idxs = torch.load(data_path)['datapoints'][clientID]
        '''else:
            self.data_idxs = range(self.file[1].size()[0])'''  
        
        # if self.split == 'fed':
            # fp=open("../data/mnist/data_distribution/training.pt", "rb")
            # data=pickle.load(fp)
            # fp.close()
            # self.data_idxs=data['datapoints'][clientID]
        # else:
            # self.data_idxs = range(self.file[1].size()[0])

    def __len__(self):
        return len(self.data_idxs)

    def __getitem__(self, index):
        '''if self.is_transform:
            image = (self.file[0][self.data_idxs[index]] - self.mean)/self.stdv
        else:
            image = self.file[0][self.data_idxs[index]]'''
        image = self.trainset[self.data_idxs[index]][0]
        #label = self.file[1][self.data_idxs[index]]
        label = self.trainset[self.data_idxs[index]][1]

        return image, label
        
